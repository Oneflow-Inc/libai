
import oneflow as flow
from oneflow import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, placement=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.weight = nn.Parameter(
            flow.empty(
                (num_embeddings, embedding_dim),
                dtype=flow.float32,
                placement=placement,
                sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
            )
        )
        nn.init.xavier_normal_(self.weight)
    
    def forward(self, input_ids):
        #   [B, B] x [S(0), B] --> [S(0), B]
        #     ↑         ↑              ↑
        #   embed    input_id       input_embed
        input_embeds = flow._C.gather(self.weight, input_ids, axis=0)
        return input_embeds


class ParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, placement=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            flow.empty(
                (num_embeddings, embedding_dim),
                dtype=flow.float32,
                placement=placement,
                sbp=[flow.sbp.broadcast, flow.sbp.split(0)],
            )
        )
        nn.init.xavier_normal_(self.weight)

    def forward(self, input_ids):
        # [B, S(0)] x [S(0), B] --> [S(0), P]
        #     ↑           ↑            ↑
        #   embed  input_ids    input_embeds
        input_embeds = flow._C.gather(self.weight, input_ids, axis=0)
        # Set the embeds sbp from [S(0), P] --> [S(0), B] to get complete embedding results.
        input_embeds = input_embeds.to_consistent(sbp=[flow.sbp.split(0), flow.sbp.broadcast])
        return input_embeds


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, placement=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(
            flow.empty(
                (in_features, out_features),
                dtype=flow.float32,
                placement=placement,
                sbp=[flow.sbp.broadcast, flow.sbp.split(1),
            )
        )
        nn.init.xavier_normal_(self.weight)

        self.bias = nn.Parameter(
            flow.zeros(
                (out_features,),
                dtype=flow.float32,
                placement=placement,
                sbp=[flow.sbp.broadcast, flow.sbp.split(0)],
            )
        )

    def forward(self, x):
        x = x.to_consistent(grad_sbp=x.sbp)
        x = flow.matmul(x, self.weight) + self.bias
        return x


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, placement=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(
            flow.empty(
                (in_features, out_features),
                dtype=flow.float32,
                placement=placement,
                sbp=[flow.sbp.broadcast, flow.sbp.split(0),
            )
        )
        nn.init.xavier_normal_(self.weight)

        self.bias = nn.Parameter(
            flow.zeros(
                (out_features,),
                dtype=flow.float32,
                placement=placement,
                sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
            )
        )

    def forward(self, x):
        x = flow.matmul(x, self.weight)
        x = x.to_consistent(sbp=[flow.sbp.split(1), flow.sbp.broadcast])
        x = x + self.bias
        return x


class MLP(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size, output_dropout_prob=0., placement=None):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, ffn_hidden_size, placement=placement)

        self.activation_func = nn.GELU()

        self.dense_4h_to_h = RowParallelLinear(ffn_hidden_size, hidden_size, placement=placement)

        self.dropout = nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        intermediate = self.dense_h_to_4h(hidden_states)
        intermediate = self.activation_func(intermediate)

        output = self.dense_4h_to_h(intermediate)
        output = self.dropout(output)
        return output


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob=0., output_dropout_prob=0., placement=None):
        super().__init__()
        self.hidden_size = hidden_size

        assert (
            hidden_size % num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads."

        self.num_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads

        self.dropout = nn.Dropout(p=attention_dropout_prob)
        self.norm_factor = 1.0 / math.sqrt(float(self.head_size))

        self.output_dropout = nn.Dropout(p=output_dropout_prob)

        self.query_key_value = ColumnParallelLinear(self.hidden_size, self.hidden_size * 3, placement=placement)

        self.dense = RowParallelLinear(self.hidden_size, self.hidden_size, placement=placement)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [S(0), B], attention_mask: [S(0), B]
        bsz, seq_len = hidden_states.size()[:2]

        query_key_value = self.query_key_value(hidden_states)
        query_key_value = query_key_value.view(bsz, -1, self.num_heads, 3 * self.head_size)
        query_key_value = query_key_value.permute(0, 2, 1, 3) # [bsz, num_heads, seq_len, 3 * head_size]
        query, key, value = flow.chunk(query_key_value, chunks=3, dim=-1)

        # [bsz, num_heads, seq_len, seq_len] with [S(0), S(1)]
        attention_scores = flow.matmul(
            query, key, transpose_b=True, alpha=self.norm_factor
        )

        # [S(0), S(1)] x [S(0), B] = [S(0), S(1)]
        attention_scores = flow.mul(attention_scores, attention_mask) - 10000.0 * (1.0 - attention_mask)
        attention_weights = flow.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Context shape: [bsz, num_heads, seq_len, head_size] with [S(0), S(1)]
        context = flow.matmul(attention_weights, value)
        context = context.permute(0, 2, 1, 3)
        context = context.view(bsz, seq_len, self.hidden_size)

        # [S(0), S(2)] x [B, S(0)] = [S(0), P] -> [S(0), B]
        output = self.dense(context)
        output = self.output_dropout(output)

        return output

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5, placement=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        self.weight = nn.Parameter(
            flow.ones(
                normalized_shape,
                dtype=flow.float32,
                placement=placement,
                sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
            )
        )

        self.bias = nn.Parameter(
            flow.zeros(
                normalized_shape,
                dtype=flow.float32,
                placement=placement,
                sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
            )
        )

    def forward(self, x):
        assert x.shape[-len(self.normalized_shape) :] == self.normalized_shape
        begin_norm_axis = x.ndim - len(self.normalized_shape)
        begin_params_axis = x.ndim - len(self.normalized_shape)
        y = flow._C.layer_norm_affine(
            x,
            self.weight,
            self.bias,
            begin_norm_axis=begin_norm_axis,
            begin_params_axis=begin_params_axis,
            epsilon=self.eps,
        )
        return y


class BertEmbeddings(nn.Module):
    def __init__(self, cfg, placement):
        super().__init__()
        self.word_embeddings = ParallelEmbedding(cfg.vocab_size, cfg.hidden_size, placement=placement)
        self.position_embeddings = Embedding(cfg.max_sequence_length, cfg.hidden_size, placement=placement)
        self.token_type_embeddings = Embedding(cfg.type_vocab_size, cfg.hidden_size, placement=placement)

        self.layernorm = LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps, placement=placement)
        self.dropout = nn.Dropout(cfg.output_dropout_prob)

        self.position_ids = flow.arange(
            cfg.max_sequence_length,
            dtype=flow.long,
            placement=placement,
            sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
        ).unsqueeze(0)
    
    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:,  :seq_length]
        
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertLayer(nn.Module):
    def __init__(self, cfg, placement):
        super().__init__()
        self.self_attention = SelfAttention(cfg.hidden_size, cfg.num_attention_heads, 
                                            attention_dropout_prob=cfg.attention_dropout_prob,
                                            output_dropout_prob=cfg.output_dropout_prob,
                                            placement=placement)   

        self.mlp = MLP(cfg.hidden_size, cfg.ffn_hidden_size, 
                       output_dropout_prob=cfg.output_dropout_prob, 
                       placement=placement)
               
        self.layernorm1 = LayerNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon, placement=placement)
        self.layernorm2 = LayerNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon, placement=placement)


    def forward(self, hidden_states, attention_mask):
        layernorm_output = self.layernorm1(hidden_states)
        attention_output = self.self_attention(layernorm_output, attention_mask)
        hidden_states = hidden_states + attention_output

        layernorm_output = self.layernorm2(hidden_states)
        mlp_output = self.mlp(layernorm_output)
        output = hidden_states + mlp_output
        
        return output
    

class BertEncoder(nn.Module):
    def __init__(self, cfg, placement):
        self.layers = nn.ModuleList(
            [BertLayer(cfg, placement=placement) for _ in range(cfg.num_layers)]
        )

    def forward(self, hidden_states, attention_mask):
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertModel(nn.Module):
    def __init__(self, cfg, placement):
        super().__init__()
        self.placement = placement
        self.embeddings = BertEmbeddings(cfg, placement=placement)
        self.encoder = BertEncoder(cfg, placement=placement)
        self.pooler = BertPooler(cfg, placement=placement)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        batch_size, seq_length = input_ids.size()
        
        if attention_mask is None:
            attention_mask = flow.ones(
                (batch_size, seq_length),
                dtype=flow.long,
                placement=placement,
                sbp=[flow.sbp.split(1), flow.sbp.broadcast],
            )

        if token_type_ids is None:
            token_type_ids = flow.zeros(
                (batch_size, seq_length),
                dtype=flow.long,
                placement=placement,
                sbp=[flow.sbp.split(1), flow.sbp.broadcast],
            )
        
        extended_attention_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length)

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_output = self.encoder(embedding_output, attention_mask=extended_attention_mask)
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output
        

class ParallelLogits(nn.Module):
    def forward(self, hidden_states, word_embeddings):
        hidden_states = hidden_states.to_consistent(grad_sbp=hidden_states.sbp)
        logits = flow.matmul(hidden_states, word_embeddings, transpose_b=True)
        return logits


class ParallelCrossEntropy(nn.Module):
    def __init__(self, ignore_index=ignore_index):
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        # Change -1 in label to 0 because sparse_softmax_cross_entropy don't accept -1
        loss_mask = labels.ne(self.ignore_index)
        labels = labels * (labels >= 0)

        loss = flow.sparse_softmax_cross_entropy(logits, labels)
        
        if loss_mask is None:
            return loss.mean()

        ntokens = loss_mask.float().sum().to_consistent(
            sbp=[flow.sbp.broadcast, flow.sbp.broadcast]
        )
        loss = flow.sum(loss * loss_mask) / ntokens
        return loss


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, cfg, placement):
        super().__init__()
        self.dense = ColumnParallelLinear(cfg.hidden_size, cfg.hidden_size, placement=placement)
        self.activation_func = nn.GELU()
        self.layernorm = LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps, placement=placement)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        hidden_states = hidden_states.to_consistent(sbp=[flow.sbp.split(0), flow.sbp.broadcast])
        hidden_states = self.layernorm(hidden_states)
        return hidden_states


class BertMLMHead(nn.Module):
    def __init__(self, cfg, placement):
        self.transform = BertPredictionHeadTransform(cfg, placement=placement)
        self.decoder = ColumnParallelLinear(cfg.hidden_size, cfg.vocab_size, placement=placement)
    
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        hidden_states = hidden_states.to_consistent(sbp=[flow.sbp.split(0), flow.sbp.broadcast])
        return hidden_states
        

class BertForMaskedLM(nn.Module):
    def __init__(self, cfg, placement):
        super().__init__()
        self.bert = BertModel(cfg, placement=placement)
        self.cls = BertMLMHead(cfg, placement=placement)
        
    def forward(self, input_id, attention_mask=None, token_type_ids=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_func = ParallelCrossEntropy(ignore_index=-1)
            loss = loss_func(logits, masked_lm_labels)
            return loss
        else:
            return logits

