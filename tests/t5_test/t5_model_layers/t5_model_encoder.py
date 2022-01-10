from libai.layers.layer_norm import LayerNorm
import oneflow as flow
from libai.layers import TransformerLayer
from libai.layers.attention import MultiheadAttention


class T5EncoderLayer(flow.nn.Module):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        scaled_init_method,
        layer_index,
        attention_dropout_prob=0.1,
        output_dropout_prob=0.0,
        layernorm_epsilon=1e-12,
        init_method=flow.nn.init.xavier_normal_,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=True,
    ) -> None:
        super().__init__()
        # Encoders
        self.layer = TransformerLayer(
            hidden_size,
            ffn_hidden_size,
            num_attention_heads,
            is_decoder=False,
            attention_dropout_prob=0.1,
            output_dropout_prob=0.0,
            layernorm_epsilon=1e-5,
            # init

            bias_gelu_fusion=bias_gelu_fusion,
            bias_dropout_fusion=bias_dropout_fusion,
            scale_mask_softmax_fusion=scale_mask_softmax_fusion,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            init_method=init_method,
            output_layer_init_method=scaled_init_method,
            layer_idx=layer_index,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        return self.layer(hidden_states, attention_mask, use_cache=use_cache)

