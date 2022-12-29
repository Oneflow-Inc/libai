import oneflow as flow
from oneflow import nn

import libai.utils.distributed as dist
from libai.config import configurable
from libai.layers import LayerNorm, LMLogits
from libai.models.utils import init_method_normal, scaled_init_method_normal
from projects.GLM.layers.embedding_layer import GLMEmbedding
from projects.GLM.layers.transformer_layer import TransformerLayer


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        layernorm_epsilon=1.0e-5,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        attention_scale=1.0,
    ):
        super().__init__()
        self.num_layers = num_layers

        def build_layer(layer_number):
            return TransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob=attention_dropout_prob,
                output_dropout_prob=output_dropout_prob,
                layernorm_epsilon=layernorm_epsilon,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                bias_gelu_fusion=bias_gelu_fusion,
                bias_dropout_fusion=bias_dropout_fusion,
                scale_mask_softmax_fusion=scale_mask_softmax_fusion,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                attention_scale=attention_scale,
                layer_idx=layer_number,
            )

        self.layers = nn.ModuleList([build_layer(i) for i in range(self.hidden_layers)])
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon, layer_idx=-1)

    def forward(self, hidden_states, attention_mask, memory_states=None):
        mem_layers = [hidden_states.detach()]
        mem_i = memory_states[i] if memory_states else None

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask, mem=mem_i)
            mem_layers.append(hidden_states.detach())

        output = self.final_layernorm(hidden_states)

        return output, mem_layers


class GPTModel(nn.Module):
    @configurable
    def __init__(
        self,
        num_layers,
        vocab_size,
        hidden_size,
        num_attention_heads,
        max_sequence_length=1024,
        embedding_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        layernorm_epsilon=1e-5,
        initializer_range=0.02,
        use_scaled_init_for_output_weights=True,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        amp_enabled=False,
        block_position_encoding=False,
        attention_scale=1.0,
        padding_idx=None,
    ):
        super().__init__()
        init_method = init_method_normal(sigma=initializer_range, mean=0)
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method_normal(initializer_range, num_layers)
        else:
            output_layer_init_method = init_method

        self.embeddings = GLMEmbedding(
            vocab_size,
            hidden_size,
            max_sequence_length,
            padding_idx=padding_idx,
            init_method=init_method,
            embedding_dropout_prob=embedding_dropout_prob,
            amp_enabled=amp_enabled,
            block_position_encoding=block_position_encoding,
        )

        self.transformer = Transformer(
            num_layers,
            hidden_size,
            num_attention_heads,
            attention_dropout_prob=attention_dropout_prob,
            output_dropout_prob=output_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            bias_gelu_fusion=bias_gelu_fusion,
            bias_dropout_fusion=bias_dropout_fusion,
            scale_mask_softmax_fusion=scale_mask_softmax_fusion,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            attention_scale=attention_scale,
        )

        self.lm_head = LMLogits(vocab_size, bias=False)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_layers": cfg.num_layers,
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "num_attention_heads": cfg.num_attention_heads,
            "max_sequence_length": cfg.max_sequence_length,
            "embedding_dropout_prob": cfg.embedding_dropout_prob,
            "attention_dropout_prob": cfg.attention_dropout_prob,
            "output_dropout_prob": cfg.output_dropout_prob,
            "layernorm_epsilon": cfg.layernorm_epsilon,
            "initializer_range": cfg.initializer_range,
            "use_scaled_init_for_output_weights": cfg.use_scaled_init_for_output_weights,
            "bias_gelu_fusion": cfg.bias_gelu_fusion,
            "bias_dropout_fusion": cfg.bias_dropout_fusion,
            "scale_mask_softmax_fusion": cfg.scale_mask_softmax_fusion,
            "apply_query_key_layer_scaling": cfg.apply_query_key_layer_scaling,
            "amp_enabled": cfg.amp_enabled,
            "block_position_encoding": cfg.block_position_encoding,
            "attention_scale": cfg.attention_scale,
            "padding_idx": cfg.padding_idx,
        }

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        memory_states=None,
        output_predict=False,
    ):
        input_ids = input_ids.to_global(placement=dist.get_layer_placement(0))
        position_ids = position_ids.to_global(placement=dist.get_layer_placement(0))
        attention_mask = attention_mask.to_global(placement=dist.get_layer_placement(0))

        batch_size, query_length = input_ids.size()
        memory_length = memory_states[0].size(1) if memory_states else 0
        is_scalar = flow.numel(attention_mask) == 1
        is_sep = is_scalar or flow.numel(attention_mask) == batch_size

        if is_sep:
            sep = attention_mask.item() if is_scalar else attention_mask
            attention_mask = self.build_mask_matrix(
                batch_size, query_length, sep, memory_length=memory_length, is_scalar=is_scalar
            )
        else:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask[:, :, :, -query_length - memory_length :]

        input_embeds = self.embeddings(input_ids, position_ids)

        logits, hidden_layers = self.transformer(
            input_embeds, attention_mask=None, memory_states=memory_states
        )
        self.update_mems(mem_layers, memory_states)

        if output_predict:
            logits = self.lm_head(logits, self.embeddings.word_embeddings.weight)

        return (logits, hidden_layers)

    def build_mask_matrix(self, batch_size, seq_length, sep, memory_length=0, is_scalar=False):
        m = flow.tril(
            flow.ones((1, seq_length, seq_length)),
        )
        if is_scalar:
            m[0, :, : int(sep)] = 1
        else:
            m = m.expand(batch_size, -1, -1)
            ids = flow.arange(seq_length, device=sep.device, dtype=sep.dtype).view(1, -1)
            mask = ids < sep.view(-1, 1)
            m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
        if memory_length > 0:
            m = m.expand(batch_size, -1, -1)
            m = flow.cat((flow.ones((batch_size, seq_length, memory_length)), m), dim=2)
        m = m.unsqueeze(1)
        m = m.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )
        return m

    def update_mems(self, hiddens, mems):
        memory_length = mems[0].size(1) if mems else 0
        query_length = hiddens[0].size(1)
        new_memory_length = memory_length + query_length

        new_mems = []
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(
                    flow.cat((mems[i][:, -new_memory_length + query_length :], hiddens[i]), dim=1)
                )
        return new_mems
