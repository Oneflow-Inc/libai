import oneflow as flow
from oneflow import nn

import libai


def cosine_similarity(x, y, dim=-1):
    return flow.sum(x * y, dim=dim) / (flow.linalg.norm(x, dim=dim) * flow.linalg.norm(y, dim=dim))


class MLPLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = libai.layers.Linear(
            cfg.hidden_size, cfg.hidden_size, bias=True, parallel="row", layer_idx=-1
        )
        self.activation = libai.layers.build_activation("tanh")

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        return x
