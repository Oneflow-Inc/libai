import oneflow as flow
from oneflow import nn

from libai.layers import Linear, build_activation


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        output_dropout_prob=0.0,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        *,
        layer_idx=0,
    ):
        super().__init__()
        self.output_dropout_prob = output_dropout_prob
        self.bias_gelu_fusion = bias_gelu_fusion
        self.bias_dropout_fusion = bias_dropout_fusion

        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.dense_h_to_4h = Linear(
            hidden_size,
            ffn_hidden_size,
            bias=False,
            parallel="col",
            skip_bias_add=bias_gelu_fusion,
            init_method=init_method,
            layer_idx=layer_idx,
        )

        if not bias_gelu_fusion:
            self.activation_func = build_activation("relu")

        self.dense_4h_to_h = Linear(
            ffn_hidden_size,
            hidden_size,
            bias=False,
            parallel="row",
            skip_bias_add=bias_dropout_fusion,
            init_method=output_layer_init_method,
            layer_idx=layer_idx,
        )

        if not bias_dropout_fusion:
            self.dropout = nn.Dropout(self.output_dropout_prob)

    def forward(self, hidden_states):
        intermediate = self.dense_h_to_4h(hidden_states)
        if self.bias_gelu_fusion:
            intermediate, bias = intermediate
            intermediate = flow._C.fused_bias_add_gelu(
                intermediate, bias, axis=intermediate.ndim - 1
            )
        else:
            intermediate = self.activation_func(intermediate)

        output = self.dense_4h_to_h(intermediate)
        if self.bias_dropout_fusion:
            output, bias = output
            output = flow._C.fused_bias_add_dropout(
                output, bias, p=self.output_dropout_prob, axis=output.ndim - 1
            )
        else:
            output = self.dropout(output)
        return output

    def extra_repr(self) -> str:
        return "bias_gelu_fusion={}, bias_dropout_fusion={}, dropout={}".format(
            self.bias_gelu_fusion, self.bias_dropout_fusion, self.output_dropout_prob
        )


class T5DenseGatedGeluDense(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        dropout_rate=0.0,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        *,
        layer_idx=0,
    ):
        super().__init__()
        self.output_dropout_prob = dropout_rate
        self.bias_gelu_fusion = bias_gelu_fusion
        self.bias_dropout_fusion = bias_dropout_fusion

        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.wi_0 = Linear(
            d_model,
            d_ff,
            bias=False,
            parallel="col",
            skip_bias_add=False,
            init_method=init_method,
            layer_idx=layer_idx,
        )

        self.wi_1 = Linear(
            d_model,
            d_ff,
            bias=False,
            parallel="row",
            skip_bias_add=False,
            init_method=init_method,
            layer_idx=layer_idx,
        )

        if not bias_gelu_fusion:
            self.activation_func = build_activation("gelu")

        self.wo = Linear(
            d_ff,
            d_model,
            bias=False,
            parallel="data",
            skip_bias_add=False,
            init_method=output_layer_init_method,
            layer_idx=layer_idx,
        )

        self.dropout = nn.Dropout(self.output_dropout_prob)

    def forward(self, hidden_states):
        wi_0_out = self.wi_0(hidden_states)
        hidden_gelu = self.activation_func(wi_0_out)
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
