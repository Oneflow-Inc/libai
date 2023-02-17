# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import oneflow as flow
import oneflow.nn.functional as F
from oneflow import nn

from libai.layers import Linear
from projects.BLOOM.modeling.activation import BloomGelu
from projects.BLOOM.modeling.attention import dropout_add


class BloomMLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        pretraining_tp,
        slow_but_exact,
        hidden_dropout,
        init_method=None,
        output_layer_init_method=None,
        layer_idx=0,
    ):
        super().__init__()
        hidden_size = hidden_size
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.pretraining_tp = pretraining_tp
        self.slow_but_exact = slow_but_exact
        self.dense_h_to_4h = Linear(
            hidden_size,
            4 * hidden_size,
            parallel="col",
            init_method=init_method,
            layer_idx=layer_idx,
        )
        self.gelu_impl = BloomGelu()
        self.dense_4h_to_h = Linear(
            4 * hidden_size,
            hidden_size,
            parallel="row",
            init_method=output_layer_init_method,
            layer_idx=layer_idx,
        )
        self.hidden_dropout = hidden_dropout

    def forward(self, hidden_states, residual):
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = flow.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + F.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)

        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

        return output
