import imp
import oneflow as flow
from oneflow import nn

# 这个是想尽可能的使用libai中封装好的layer，需要看看是否transform层可以利用

# 有待完成的工作：
#   1. 看看是否可以将Transformer层进行利用
#   2. 修改mask层的实现

from libai.layers import(
    Embedding,
    VocabEmbedding,
    LayerNorm,
    TransformerLayer
)

from .roberta_utils import (
    init_weights,
    create_position_ids_from_input_ids,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
    apply_chunking_to_forward,
    position_scores  # replace einsum
)

class RobertaEmbeddings(nn.Module):

    def __init__(
        self, 
        vocab_size, 
        max_position_embeddings, 
        type_vocab_size,
        hidden_size,
        layer_norm_eps=1e-5, 
        dropout=0, 
        pad_token_id=0,
        position_embedding_type="absolute"
    ):
        super(RobertaEmbeddings, self).__init__()
        # embedding
        self.word_embeddings = VocabEmbedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = position_embedding_type
        self.register_buffer("position_ids", flow.arange(max_position_embeddings).expand(1, -1))
        self.register_buffer("token_type_ids", flow.zeros(self.position_ids.size(), dtype=flow.int64, device=self.position_ids.device), persistent=False,)

        self.padding_idx = pad_token_id

    def forward(
        self, 
        input_ids, 
        token_type_ids=None, 
        position_ids=None, 
        inputs_embeds=None, 
        past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        if token_type_ids is None:
            buffered_token_type_ids = self.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                input_shape[0], seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 这里比bert增加了一个layernorm
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: flow.Tensor

        Returns: flow.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = flow.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=flow.int64, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

class RobertaExtendedAttnMask(nn.Module):
    """
    有待实现
    """
    
class RobertaPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class RoBERTaModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        layer_norm_eps, 
        hidden_dropout, 
        pad_token_id, 
        position_embedding_type,
        max_position_embeddings, 
        type_vocab_size,
        hidden_layers,
    ):
        super().__init__()
        init_method = init_method_normal(initializer_range)
        scaled_init_method = scaled_init_method_normal(initializer_range, hidden_layers)

        # Embeddings
        self.embeddings = RobertaEmbeddings(
            vocab_size, 
            max_position_embeddings, 
            type_vocab_size,
            hidden_size, 
            layer_norm_eps, 
            hidden_dropout, 
            pad_token_id, 
            position_embedding_type
        )

        # Mask generation
        # 这里应该需要改成roberta自己的可以生成attention的方法
        self.extended_attn_mask = RobertaExtendedAttnMask()

        # Encoders
        self.encoders = nn.ModuleList(
            [
                TransformerLayer(
                    
                )
                for i in range(hidden_layers)
            ]
        )
        self.final_layernorm = LayerNorm((hidden_size,), eps=layernorm_eps, layer_idx=-1)

        self.pooler = RobertaPooler(hidden_size, init_method) if add_pooling_layer else None

    @classmethod
    def from_config(cls, cfg):
        return {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "hidden_layers": cfg.hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "intermediate_size": cfg.intermediate_size,
            "hidden_dropout_prob": cfg.hidden_dropout_prob,
            "attention_probs_dropout_prob": cfg.attention_probs_dropout_prob,
            "max_position_embeddings": cfg.max_position_embeddings,
            "num_tokentypes": cfg.num_tokentypes,
            "add_pooling_layer": cfg.add_pooling_layer,
            "initializer_range": cfg.initializer_range,
            "layernorm_eps": cfg.layernorm_eps,
            "bias_gelu_fusion": cfg.bias_gelu_fusion,
            "bias_dropout_fusion": cfg.bias_dropout_fusion,
            "scale_mask_softmax_fusion": cfg.scale_mask_softmax_fusion,
            "apply_query_key_layer_scaling": cfg.apply_query_key_layer_scaling,
        }

    def forward(self, input_ids, attention_mask, tokentype_ids=None):
        extended_attention_mask = self.extended_attn_mask(attention_mask)
        embedding_output = self.embeddings(input_ids, tokentype_ids)

        hidden_states = embedding_output
        for layer in self.encoders:
            hidden_states = layer(hidden_states, extended_attention_mask)
        encoder_output = self.final_layernorm(hidden_states)
        pooled_output = self.pooler(encoder_output) if self.pooler is not None else None
        return encoder_output, pooled_output

    def word_embeddings_weight(self):
        return self.embeddings.word_embeddings()
