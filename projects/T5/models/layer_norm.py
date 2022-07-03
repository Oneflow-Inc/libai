import oneflow as flow
from libai.utils import distributed as dist


class LayerNorm(flow.nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.weight = flow.nn.Parameter(
                flow.ones(
                    normalized_shape,
                    dtype=flow.float32,
                    placement=dist.get_layer_placement(layer_idx),
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                )
            )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(flow.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * flow.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [flow.float16, flow.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states