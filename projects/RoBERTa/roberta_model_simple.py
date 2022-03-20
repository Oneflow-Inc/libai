import oneflow as flow
from oneflow import nn

from libai.config import configurable
from libai.layers import (
    Embedding,
    LayerNorm,
    Linear,
    LMLogits,
    ParallelCrossEntropyLoss,
    TransformerLayer,
    VocabEmbedding,
    build_activation,
)
from libai.utils import distributed as dist

from libai.models.build import MODEL_ARCH_REGISTRY
from libai.models.utils import init_method_normal, scaled_init_method_normal

class RoBERTaEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_position_embeddings,
        type_vocab_size,
        hidden_size,
        layer_norm_eps=1e-5,
        embedding_dropout_prob=0,
        pad_token_id=0,
        position_embedding_type="absolute",
        init_method = nn.init.xavier_normal_,
        amp_enabled=False,                                              # bert中的这个是干什么的
    ) -> None:
        super(RoBERTaEmbeddings, self).__init__()
        self.word_embeddings = VocabEmbedding(
            vocab_size, hidden_size, padding_idx=pad_token_id,init_method=init_method, amp_enabled=amp_enabled
        )
        self.position_embeddings = Embedding(
            max_position_embeddings, hidden_size, init_method=init_method, amp_enabled=amp_enabled
        )
        # 对应原始论文中的segment_id
        self.token_type_embeddings = Embedding(
            type_vocab_size, hidden_size, init_method=init_method, amp_enabled=amp_enabled
        )

        self.position_embedding_type = position_embedding_type
        self.position_id = flow.arange(
            max_position_embeddings,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        ).unsqueeze(0)

        self.token_type_ids = flow.zeros(
            self.position_ids.size(), dtype=flow.int64,sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0)
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout_prob)
        self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.padding_idx = pad_token_id


    def forward(
        self, 
        input_ids, 
        token_type_ids=None, 
        position_ids=None, 
        inputs_embeds=None, 
        past_key_values_length=0
    ):
        seq_length = input_ids.size()[1]

        # word_embeddings
        word_embeddings = self.word_embeddings(input_ids)
        # position_embeddings
        if position_ids is None:
            position_ids = (
                self.position_ids[:, :seq_length].expand_as(input_ids).to_global(sbp=input_ids.sbp)
            )
        position_embeddings = self.position_embeddings(position_ids)
        # token_type_embeddings
        if token_type_ids is None:
            tokentype_ids = (
                    self.tokentype_ids[:, :seq_length].expand_as(input_ids).to_global(sbp=input_ids.sbp)
            )
        # embeddings
        embeddings = word_embeddings
        if self.position_embedding_type == "absolute":
            embeddings +=position_embeddings         
        embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        return embeddings

    def word_embeddings(self):
        return self.word_embeddings.weight

class RoBERTaEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class RoBERTaPooler(nn.Module):
    def __init__(self, hidden_size, init_method):
        super().__init__()
        self.dense = Linear(hidden_size, hidden_size, init_method=init_method, layer_idx=-1,)
        self.activation = build_activation("tanh")

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class RoBERTaExtendedAttnMask(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class RoBERTaModel(nn.Module):
    @configurable
    def __init__(self, args)-> None:
        super().__init__()
        # init param
        init_method = init_method_normal(args.initializer_range)
        scaled_init_method = scaled_init_method_normal(args.initializer_range, args.hidden_layers)

        # embeddings
        self.embeddings = RoBERTaEmbeddings(
            args.vocab_size,
            args.max_position_embeddings,
            args.type_vocab_size,
            args.hidden_size,
            args.layer_norm_eps,
            args.embedding_dropout_prob,
            args.pad_token_id,
            args.position_embedding_type,
        )

        # mask
        self.extended_atten_mask = RoBERTaExtendedAttnMask()

        # encoder
        # roberta和bert在encoder部分应该是相同的~
        self.encoder = nn.ModuleList(
            [
                TransformerLayer(
                    args.hidden_size,
                    args.intermediate_size,
                    args.num_attention_heads,
                    attention_dropout_prob=args.attention_probs_dropout_prob,
                    output_dropout_prob=args.hidden_dropout_prob,
                    layernorm_epsilon=args.layernorm_eps,
                    bias_gelu_fusion=args.bias_gelu_fusion,
                    bias_dropout_fusion=args.bias_dropout_fusion,
                    scale_mask_softmax_fusion=args.scale_mask_softmax_fusion,
                    apply_query_key_layer_scaling=args.apply_query_key_layer_scaling,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method,
                    layer_idx=i,
                    is_deocder=False,
                )
                for i in range(args.hidden_layers)
            ]
        )

        # 最后的pooler层是什么作用啊
        self.pooler = RoBERTaPooler(args.hidden_size, init_method)


    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
    ):

        if input_ids is None:
            raise ValueError(
                "The input_ids cannot be empty!"
            )
        
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        # 开始获取新的mask
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if attention_mask is None:
            attention_mask = flow.ones(
                ((batch_size, seq_length + past_key_values_length)), 
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(0)
            )

        
        

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: flow.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape )
            # attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = flow.ones(
                    encoder_hidden_shape, sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),placement=dist.get_layer_placement(0))
                    # encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_layers x num_heads]
        # and head_mask is converted to shape [num_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        embedding_output = self.embeddings(
            input_ids, token_type_ids, position_ids, inputs_embeds, past_key_values_length)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask, encoder_hidden_states,
                                       encoder_extended_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(
            sequence_output) if self.pooler is not None else None

        # sequence_output, pooled_output, past_key_values, hidden_states, attentions, cross_attentions.
        # return (sequence_output, pooled_output) + encoder_outputs[1:]
        return encoder_outputs, pooled_output



