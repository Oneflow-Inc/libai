"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow as flow
import oneflow.nn as nn


class Interaction(nn.Module):
    def __init__(
        self,
        dense_feature_size,
        num_embedding_fields,
        interaction_itself=False,
        interaction_padding=True,
    ):
        super(Interaction, self).__init__()
        self.interaction_itself = interaction_itself
        n_cols = num_embedding_fields + 2 if self.interaction_itself else num_embedding_fields + 1
        output_size = dense_feature_size + sum(range(n_cols))
        self.output_size = ((output_size + 8 - 1) // 8 * 8) if interaction_padding else output_size
        self.output_padding = self.output_size - output_size

    def forward(self, x: flow.Tensor, y: flow.Tensor) -> flow.Tensor:
        (bsz, d) = x.shape
        return flow._C.fused_dot_feature_interaction(
            [x.view(bsz, 1, d), y],
            output_concat=x,
            self_interaction=self.interaction_itself,
            output_padding=self.output_padding,
        )
