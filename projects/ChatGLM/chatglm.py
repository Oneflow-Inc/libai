# --------------------------------------------------------
# Refer to code from:
# https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py
# --------------------------------------------------------

import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import oneflow as flow
from oneflow import nn

from libai.inference.generator.generation_utils import Generator, LogitsProcessorList
from libai.layers import LayerNorm, Linear, RMSLayerNorm, VocabEmbedding
from libai.utils import distributed as dist


def apply_rotary_pos_emb(x: flow.Tensor, rope_cache: flow.Tensor) -> flow.Tensor:
    # x: [sq, b, np, hn]
    sq, _, np, _ = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = flow.cat(
        [
            (xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1]).unsqueeze(
                -1
            ),
            (xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1]).unsqueeze(
                -1
            ),
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return flow.cat((x_out2, x_pass), dim=-1)


class PrefixEncoder(flow.nn.Module):
    """
    encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, cfg):
        super().__init__()
        self.prefix_projection = cfg.prefix_projection
        kv_size = cfg.num_layers * cfg.kv_channels * cfg.multi_query_group_num * 2
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix

            self.embedding = VocabEmbedding(cfg.pre_seq_len, kv_size, amp_enabled=cfg.amp_enabled)
            self.trans = nn.Sequential(
                Linear(kv_size, cfg.hidden_size, parallel="col"),
                nn.Tanh(),
                Linear(cfg.hidden_size, kv_size, parallel="row"),
            )
        else:
            self.embedding = VocabEmbedding(cfg.pre_seq_len, kv_size, amp_enabled=cfg.amp_enabled)

    def forward(self, prefix):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_length,
        original_impl=False,
        layer_idx=0,
    ):
        super().__init__()
        sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        placement = dist.get_layer_placement(layer_idx)
        theta = 1.0 / (
            10000
            ** (flow.arange(0, dim, 2, dtype=flow.float32, placement=placement, sbp=sbp) / dim)
        )
        seq_idx = flow.arange(max_length, dtype=flow.float32, placement=placement, sbp=sbp)
        idx_theta = flow.matmul(seq_idx.unsqueeze(1), theta.unsqueeze(0)).float()
        self.sin_cache = flow.sin(idx_theta)
        self.cos_cache = flow.cos(idx_theta)
        self.max_length = max_length

    def forward_impl(self, seq_len: int, dtype: flow.dtype):
        """Enhanced Transformer with Rotary Position Embedding.
        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/
        master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        assert seq_len <= self.max_length
        cache = flow.stack([self.cos_cache[:seq_len], self.sin_cache[:seq_len]], dim=-1)
        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (flow.float16, flow.bfloat16, flow.int8):
            cache = cache.bfloat16() if dtype == flow.bfloat16 else cache.half()
        return cache

    def forward(
        self,
        max_seq_len,
        dtype,
        offset=0,
    ):
        return self.forward_impl(max_seq_len, dtype=dtype)


class CoreAttention(flow.nn.Module):
    def __init__(self, cfg, layer_number):
        super(CoreAttention, self).__init__()

        self.apply_query_key_layer_scaling = cfg.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = cfg.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        projection_size = cfg.kv_channels * cfg.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // cfg.num_attention_heads
        self.num_attention_heads_per_partition = cfg.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = nn.Dropout(cfg.attention_dropout)

    def scaled_dot_product_attention(
        self, query, key, value, attn_mask=None, is_causal=False, dropout_p=0.0
    ):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = flow.zeros(
            L,
            S,
            dtype=query.dtype,
            placement=query.placement,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        )

        if is_causal:
            assert attn_mask is None
            temp_mask = flow.ones(
                L,
                S,
                dtype=flow.bool,
                placement=query.placement,
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            ).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == flow.bool:
                attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = flow.matmul(query, key, transpose_b=True, alpha=scale_factor)
        attn_weight += attn_bias
        attn_weight = flow.softmax(attn_weight, dim=-1)
        ans = flow.matmul(attn_weight, value)
        return ans

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None):
        # query_layer: [sq, b, np, hn] -[premute]-> [batch_size, head_num, seq_len, hidden_size]
        query_layer, key_layer, value_layer = [
            k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]
        ]
        if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
            context_layer = self.scaled_dot_product_attention(
                query_layer, key_layer, value_layer, is_causal=True
            )

        else:
            if attention_mask is not None:
                attention_mask = ~attention_mask
            context_layer = self.scaled_dot_product_attention(
                query_layer, key_layer, value_layer, attention_mask
            )

        context_layer = context_layer.permute(2, 0, 1, 3)
        context_layer = context_layer.flatten(2)
        return context_layer


def split_tensor_along_last_dim(
    tensor: flow.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[flow.Tensor]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = flow.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class SelfAttention(flow.nn.Module):
    """Parallel self-attention layer abstract class.
    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, cfg, layer_number):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)

        self.projection_size = cfg.kv_channels * cfg.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // cfg.num_attention_heads
        self.num_attention_heads_per_partition = cfg.num_attention_heads

        self.multi_query_attention = cfg.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = cfg.multi_query_group_num
            self.qkv_hidden_size = (
                self.projection_size
                + 2 * self.hidden_size_per_attention_head * cfg.multi_query_group_num
            )

        self.query_key_value = Linear(
            cfg.hidden_size,
            self.qkv_hidden_size,
            bias=cfg.add_bias_linear or cfg.add_qkv_bias,
            parallel="col",
            layer_idx=self.layer_number - 1,
        )

        self.core_attention = CoreAttention(cfg, self.layer_number)

        # Output.
        self.dense = Linear(
            self.projection_size,
            cfg.hidden_size,
            bias=cfg.add_bias_linear,
            parallel="col",
            layer_idx=self.layer_number - 1,
        )

    def forward(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1]
                + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )  # [sq, b, num_attention_heads_per_partition,hidden_size_per_attention_head]
            key_layer = key_layer.view(
                key_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )  # [sq, b, num_multi_query_groups_per_partition,hidden_size_per_attention_head]
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )  # [sq, b, num_multi_query_groups_per_partition,hidden_size_per_attention_head]

        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = flow.chunk(mixed_x_layer, 3, dim=-1)

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = flow.cat((cache_k, key_layer), dim=0)
            value_layer = flow.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(
                -1,
                -1,
                -1,
                self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition,
                -1,
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2]
                + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1,
                -1,
                -1,
                self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition,
                -1,
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2]
                + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        return output, kv_cache


class MLP(flow.nn.Module):
    """MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, cfg, layer_idx):
        super(MLP, self).__init__()

        self.add_bias = cfg.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see
        # https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = Linear(
            cfg.hidden_size,
            cfg.ffn_hidden_size * 2,
            bias=self.add_bias,
            parallel="col",
            layer_idx=layer_idx,
        )

        def swiglu(x):
            x = flow.chunk(x, 2, dim=-1)
            return nn.functional.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = Linear(
            cfg.ffn_hidden_size,
            cfg.hidden_size,
            bias=self.add_bias,
            parallel="row",
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(flow.nn.Module):
    """A single transformer layer.
    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, cfg, layer_number):
        super(GLMBlock, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = cfg.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = cfg.fp32_residual_connection

        LayerNormFunc = RMSLayerNorm if cfg.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(
            (cfg.hidden_size,), eps=cfg.layernorm_epsilon, layer_idx=layer_number - 1
        )

        # Self attention.
        self.self_attention = SelfAttention(cfg, layer_number)
        self.hidden_dropout = cfg.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(
            (cfg.hidden_size,), eps=cfg.layernorm_epsilon, layer_idx=layer_number - 1
        )

        # MLP
        self.mlp = MLP(cfg, layer_number - 1)

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
    ):
        # hidden_states: [s, b, h]
        hidden_states = hidden_states.to_global(
            placement=dist.get_layer_placement(self.layer_number - 1)
        )

        if attention_mask is not None:
            attention_mask = attention_mask.to_global(
                placement=dist.get_layer_placement(self.layer_number - 1)
            )
        if rotary_pos_emb is not None:
            rotary_pos_emb = rotary_pos_emb.to_global(
                placement=dist.get_layer_placement(self.layer_number - 1),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output, attention_mask, rotary_pos_emb, kv_cache=kv_cache, use_cache=use_cache
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = nn.functional.dropout(
            attention_output, p=self.hidden_dropout, training=self.training
        )
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output, kv_cache


class GLMTransformer(flow.nn.Module):
    """Transformer class."""

    def __init__(self, cfg):
        super(GLMTransformer, self).__init__()

        self.fp32_residual_connection = cfg.fp32_residual_connection
        self.post_layer_norm = cfg.post_layer_norm

        # Number of layers.
        self.num_layers = cfg.num_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(cfg, layer_number)

        self.layers = flow.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSLayerNorm if cfg.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(
                (cfg.hidden_size,), eps=cfg.layernorm_epsilon, layer_idx=-1
            )

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_caches=None,
        use_cache: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
    ):
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)

            layer_ret = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_cache=kv_caches[index],
                use_cache=use_cache,
            )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions


class ChatGLMPreTrainedModel(nn.Module):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        return

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        full_attention_mask = flow.ones(
            batch_size,
            seq_length,
            seq_length,
            placement=dist.get_layer_placement(0),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        )
        full_attention_mask = full_attention_mask.tril()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        if past_length:
            full_attention_mask = flow.cat(
                (
                    flow.ones(
                        batch_size,
                        seq_length,
                        past_length,
                        placement=dist.get_layer_placement(0),
                        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                    ),
                    full_attention_mask,
                ),
                dim=-1,
            )
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask = full_attention_mask.unsqueeze(1)
        return full_attention_mask

    def get_position_ids(self, input_ids):
        batch_size, seq_length = input_ids.shape
        position_ids = (
            flow.arange(
                seq_length,
                dtype=flow.long,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        return position_ids


class EmbeddingLayer(flow.nn.Module):
    """Language model embeddings."""

    def __init__(self, cfg):
        super(EmbeddingLayer, self).__init__()

        self.hidden_size = cfg.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = VocabEmbedding(
            cfg.padded_vocab_size, self.hidden_size, amp_enabled=cfg.amp_enabled
        )
        self.fp32_residual_connection = cfg.fp32_residual_connection

    def forward(self, input_ids):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class ChatGLMModel(ChatGLMPreTrainedModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = EmbeddingLayer(cfg)
        self.num_layers = cfg.num_layers
        self.multi_query_group_num = cfg.multi_query_group_num
        self.kv_channels = cfg.kv_channels

        # Rotary positional embeddings
        self.seq_length = cfg.seq_length
        rotary_dim = (
            cfg.hidden_size // cfg.num_attention_heads
            if cfg.kv_channels is None
            else cfg.kv_channels
        )
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, cfg.seq_length)

        self.encoder = GLMTransformer(cfg)
        self.output_layer = Linear(cfg.hidden_size, cfg.padded_vocab_size, bias=False, layer_idx=-1)
        self.pre_seq_len = cfg.pre_seq_len
        self.prefix_projection = cfg.prefix_projection
        if self.pre_seq_len is not None:
            for param in self.parameters():
                param.requires_grad = False
            self.prefix_tokens = flow.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(cfg)
            self.dropout = flow.nn.Dropout(0.1)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_layers * 2,
            self.multi_query_group_num,
            self.kv_channels,
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids,
        position_ids: Optional[flow.Tensor] = None,
        attention_mask: Optional[flow.BoolTensor] = None,
        full_attention_mask: Optional[flow.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[flow.Tensor, flow.Tensor], ...]] = None,
        inputs_embeds: Optional[flow.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.cfg.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.cfg.use_cache
        return_dict = return_dict if return_dict is not None else self.cfg.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size)
            if attention_mask is not None:
                attention_mask = flow.cat(
                    [flow.ones((batch_size, self.pre_seq_len)), attention_mask], dim=-1
                )

        if full_attention_mask is None:
            if past_key_values and seq_length != 1:
                full_attention_mask = self.get_masks(
                    input_ids, past_key_values, padding_mask=attention_mask
                )

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length, inputs_embeds.dtype)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        # rotary_pos_emb = None

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return dict(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel, Generator):
    def __init__(self, cfg):
        super().__init__()

        self.max_sequence_length = cfg.max_length
        self.transformer = ChatGLMModel(cfg)
        self.cfg = cfg
        self.loss_fct = nn.CrossEntropyLoss()

    def _update_model_kwargs_for_generation(
        self,
        outputs: Dict,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = outputs["past_key_values"]

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = flow.cat(
                [
                    attention_mask,
                    attention_mask.new_ones(
                        (attention_mask.shape[0], 1),
                        sbp=attention_mask.sbp,
                        placement=attention_mask.placement,
                    ),
                ],
                dim=-1,
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = flow.cat([position_ids, new_position_id], dim=-1)

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: flow.LongTensor,
        past_key_values: Optional[flow.Tensor] = None,
        attention_mask: Optional[flow.Tensor] = None,
        position_ids: Optional[flow.Tensor] = None,
        use_cache: Optional[bool] = None,
        is_first_forward: bool = True,
        **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids)
        if not is_first_forward:
            if past_key_values is not None:
                position_ids = position_ids[..., -1:]
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids: Optional[flow.Tensor] = None,
        position_ids: Optional[flow.Tensor] = None,
        attention_mask: Optional[flow.Tensor] = None,
        past_key_values: Optional[Tuple[flow.FloatTensor]] = None,
        inputs_embeds: Optional[flow.Tensor] = None,
        labels: Optional[flow.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
    ):
        use_cache = use_cache if use_cache is not None else self.cfg.use_cache
        return_dict = return_dict if return_dict is not None else self.cfg.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = transformer_outputs["last_hidden_state"]
        if return_last_logit:
            hidden_states = hidden_states[-1:]
        lm_logits = self.transformer.output_layer(hidden_states)
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        loss = None
        if labels is not None:

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            # loss = loss.mean()

        if labels is not None:
            return dict(
                loss=loss,
            )
        else:
            return dict(logits=lm_logits, past_key_values=transformer_outputs["past_key_values"])

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[flow.Tensor, flow.Tensor], ...], beam_idx: flow.LongTensor
    ) -> Tuple[Tuple[flow.Tensor, flow.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if
        [`~PreTrainedModel.beam_search`] or [`~PreTrainedModel.beam_sample`]
        is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx),
                layer_past[1].index_select(1, beam_idx),
            )
            for layer_past in past
        )

    def process_response(self, output, history):
        content = ""
        history = deepcopy(history)
        for response in output.split("<|assistant|>"):
            if "\n" in response:
                metadata, content = response.split("\n", maxsplit=1)
            else:
                metadata, content = "", response
            if not metadata.strip():
                content = content.strip()
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                content = content.replace("[[训练时间]]", "2023年")
            else:
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                if history[0]["role"] == "system" and "tools" in history[0]:
                    content = "\n".join(content.split("\n")[1:-1])

                    def tool_call(**kwargs):
                        return kwargs

                    parameters = eval(content)
                    content = {"name": metadata.strip(), "parameters": parameters}
                else:
                    content = {"name": metadata.strip(), "content": content}
        return content, history

    @flow.inference_mode()
    def chat(
        self,
        tokenizer,
        query: str,
        history: List[Dict] = None,
        role: str = "user",
        max_length: int = 8192,
        num_beams=1,
        do_sample=True,
        top_p=0.8,
        temperature=0.8,
        logits_processor=LogitsProcessorList(),
        **kwargs,
    ):
        if history is None:
            history = []

        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            **kwargs,
        }
        inputs = tokenizer.build_chat_input(query, history=history, role=role)
        inputs = tokenizer.convert_to_tensors(inputs, return_tensors="of", is_global=True)
        outputs = self.generate(inputs, **gen_kwargs, eos_token_id=tokenizer.eos_token_id)
        outputs = outputs.tolist()[0][inputs.size(1) : -1]
        response = tokenizer.decode(outputs)
        history.append({"role": role, "content": query})
        response, history = self.process_response(response, history)
        return response, history

    @staticmethod
    def set_activation_checkpoint(model):
        for module_block in model.modules():
            # Old API in OneFlow 0.8
            if hasattr(module_block, "origin"):
                if isinstance(module_block.origin, GLMBlock):
                    module_block.cfg.activation_checkpointing = True
            else:
                if isinstance(module_block.to(nn.Module), GLMBlock):
                    module_block.to(nn.graph.GraphModule).activation_checkpointing = True
