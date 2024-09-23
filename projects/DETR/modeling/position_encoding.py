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


# --------------------------------------------------------
# DETR Model
# References:
# https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
# --------------------------------------------------------


import math

import oneflow as flow
import oneflow.nn as nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        _, mask = tensor_list
        assert mask is not None
        not_mask = ~mask

        y_embed = flow.cumsum(not_mask, dim=1)
        x_embed = flow.cumsum(not_mask, dim=2)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = flow.arange(self.num_pos_feats, dtype=flow.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        if x_embed.is_global:
            pos_x = x_embed[:, :, :, None] / dim_t.to_global(
                sbp=x_embed.sbp, placement=x_embed.placement
            )
            pos_y = y_embed[:, :, :, None] / dim_t.to_global(
                sbp=y_embed.sbp, placement=y_embed.placement
            )
        else:
            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = flow.stack((pos_x[:, :, :, 0::2].sin(), 
                            pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = flow.stack((pos_y[:, :, :, 0::2].sin(), 
                            pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = flow.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.to_global(sbp=flow.sbp.split(1), placement=pos.placement)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x, _ = tensor_list
        h, w = x.shape[-2:]
        if x.is_global:
            i = flow.arange(w).to_global(sbp=flow.sbp.broadcast, placement=x.placement)
            j = flow.arange(h).to_global(sbp=flow.sbp.broadcast, placement=x.placement)
        else:
            i = flow.arange(w)
            j = flow.arange(h)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            flow.cat(
                [x_emb.unsqueeze(0).repeat(h, 1, 1), y_emb.unsqueeze(1).repeat(1, w, 1)], dim=-1
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )
        return pos.to_global(sbp=flow.sbp.split(1), placement=pos.placement)
