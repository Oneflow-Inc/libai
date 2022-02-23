import oneflow as flow


class LayerNorm(flow.nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = flow.nn.Parameter(flow.Tensor(
            flow.ones(features, dtype=flow.float32)))
        self.b_2 = flow.nn.Parameter(flow.Tensor(
            flow.zeros(features, dtype=flow.float32)))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)

        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
