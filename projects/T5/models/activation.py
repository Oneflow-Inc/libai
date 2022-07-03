import math
from enum import Enum
from typing import Optional

import oneflow as flow
from oneflow import nn


class Activation(str, Enum):
    SquaredReLU = "squared_relu"
    NewGELU = "new_glue"
    GeLU = "gelu"
    LeakyReLU = "leaky_relu"
    ReLU = "relu"
    Tanh = "tanh"


class NewGELU(nn.Module):
    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return 0.5 * x * (1.0 + flow.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * flow.pow(x, 3.0))))


class SquaredReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x_ = flow._C.relu(x)
        return x_ * x_


class Passthrough(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return x


def build_activation(activation: Optional[Activation]):
    """
    Fetching activation layers by name, e.g.,
    ``build_activation("gelu")`` returns ``nn.GELU()`` module.
    """
    if not activation:
        return Passthrough()

    return {
        Activation.ReLU: nn.ReLU,
        Activation.GeLU: nn.GELU,
        Activation.LeakyReLU: nn.LeakyReLU,
        Activation.SquaredReLU: SquaredReLU,
        Activation.Tanh: nn.Tanh,
        Activation.NewGELU: NewGELU,
    }[activation]()
