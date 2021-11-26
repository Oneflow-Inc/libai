# -*- coding: utf-8 -*-
# Copyright (c) OneFlow, Inc. and its affiliates.

from enum import Enum
from typing import Optional

import oneflow as flow
from oneflow import nn


class Activation(str, Enum):
    SquaredReLU = "squared_relu"
    GeLU = "gelu"
    LeakyReLU = "leaky_relu"
    ReLU = "relu"
    Tanh = "tanh"


# For unit testing / parity comparisons, probably not the fastest way
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
    if not activation:
        return Passthrough()

    return {
        Activation.ReLU: nn.ReLU,
        Activation.GeLU: nn.GELU,
        Activation.LeakyReLU: nn.LeakyReLU,
        Activation.SquaredReLU: SquaredReLU,
        Activation.Tanh: nn.Tanh,
    }[activation]()
