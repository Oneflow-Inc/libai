# coding=utf-8
"""
Copyright 2021 The OneFlow Authors. All rights reserved.
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
from oneflow import nn

from libai.utils import distributed as dist


class TokenTypeEmbedding(nn.Module):
    """Construct the token_type embeddings.
    """

    def __init__(
        self, num_tokentypes, hidden_size, init_method,
    ):
        super().__init__()
        self.init_method = init_method

        assert num_tokentypes > 0, ""
        self.tokentype_embeddings = nn.Parameter(
            flow.empty(
                (num_tokentypes, hidden_size),
                dtype=flow.float32,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        self.init_method(self.tokentype_embeddings)

    def forward(self, tokentype_ids):
        tokentype_embeds = flow._C.gather(
            self.tokentype_embeddings, tokentype_ids, axis=0
        )
        return tokentype_embeds
