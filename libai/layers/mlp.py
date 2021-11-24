# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
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
import oneflow.nn.init as init
from libai import distribute as dist

from .linear import ColumnParallelLinear, RowParallelLinear

class ParallelMLP(flow.nn.Module):
    """
    ParallelMLP will take the input with h hidden state, project it to 4 * h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.

    Arguments:
        layer_idx: the layer index, which determines the placement.
        hidden_size: size of hidden state.
        output_dropout_prob: dropout probability of output.
        init_method: method to initialize the input layer weights.
        output_layer_init_method: method to initialize the output layer weights. If None, use `init_method`.
        bias_gelu_fusion: whether fuse add bias and gelu.
        bias_dropout_fusion: whether fuse add bias and dropout.
    """
    def __init__(self, layer_idx, hidden_size, output_dropout_prob, init_method, output_layer_init_method=None, bias_gelu_fusion=False, bias_dropout_fusion=False):
        super().__init__()
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.c_fc = ColumnParallelLinear(layer_idx, hidden_size, hidden_size * 4, init_method=init_method, 
                                         need_gelu=True, bias_gelu_fusion=bias_gelu_fusion)
        self.c_proj = RowParallelLinear(layer_idx, hidden_size * 4, hidden_size, init_method=output_layer_init_method, 
                                        output_dropout_prob=output_dropout_prob, bias_dropout_fusion=bias_dropout_fusion)

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # sbp: [S(0), B]
        h = self.c_fc(hidden_states)
        h = self.c_proj(h)
        return h

