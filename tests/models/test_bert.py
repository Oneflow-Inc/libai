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

from omegaconf import DictConfig
from libai.models import build_model

cfg = dict(
    model_name = "BertForPreTraining",
    model_cfg = dict(
        vocab_size=1000,
        hidden_size=768,
        hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        num_tokentypes=2,
        add_pooling_layer=True,
        initializer_range=0.02,
        layernorm_eps=1e-12,
        bias_gelu_fusion=True,
        bias_dropout_fusion=True,
        scale_mask_softmax_fusion=True,
        apply_query_key_layer_scaling=True,
    )
)

cfg = DictConfig(cfg)

# tests build_model
bert_model = build_model(cfg)