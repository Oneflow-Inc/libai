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

from omegaconf import OmegaConf

from libai.models import build_model

cfg = OmegaConf.create()

cfg.model_name = "VisionTransformer"
cfg.model_cfg = dict(
        img_size=224,
        patch_size=16,
        hidden_dim=768,
        mlp_dim=3072,
        num_heads=12,
        num_layers=12,
        num_classes=1000,
        attn_dropout=0.0,
        dropout=0.1,
    )

# test build_model
vit_b_16_224 = build_model(cfg)