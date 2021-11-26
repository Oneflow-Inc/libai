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


from oneflow import nn

from libai.layers import Linear1D


class MLP(nn.Module):
    """ MLP

    MLP will take the input with h hidden state, project it to intermediate 
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension.
    
    Arguments:
        hidden_size: size of each input and output sample.
        ffn_hidden_size: size of each intermediate sample.
        output_dropout_prob: Output dropout probability. Defaults to 0.0.
        init_method: method to initialize the first linear weight. Defaults to nn.init.xavier_normal_.
        output_layer_init_method: method to initialize the second linear weight. Defaults to nn.init.xavier_normal_.
        layer_idx: A layer_idx sign which determines the placement. It will be used in pipeline parallelism. Defaults to 0.
    """

    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        output_dropout_prob=0.0,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=nn.init.xavier_normal_,
        *,
        layer_idx=0,
    ):
        super().__init__()

        self.dense_h_to_4h = Linear1D(
            hidden_size,
            ffn_hidden_size,
            bias=True,
            parallel="col",
            activation="gelu",
            bias_gelu_fusion=True,
            init_method=init_method,
            layer_idx=layer_idx,
        )
        self.dense_4h_to_h = Linear1D(
            ffn_hidden_size,
            hidden_size,
            bias=True,
            parallel="row",
            output_dropout_prob=output_dropout_prob,
            bias_dropout_fusion=True,
            init_method=output_layer_init_method,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states):
        intermediate = self.dense_h_to_4h(hidden_states)
        output = self.dense_4h_to_h(intermediate)
        return output
