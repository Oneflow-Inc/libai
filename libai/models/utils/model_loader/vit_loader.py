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

import json

import oneflow as flow

from .base_loader import ModelLoaderHuggerFace, ModelLoaderLiBai


class ViTLoaderHuggerFace(ModelLoaderHuggerFace):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)

        """NOTE: base_model_prefix_1 is ViT's prefix in Transformers.
        base_model_prefix_2 is ViT's prefix in LiBai."""

        self.base_model_prefix_1 = "vit"
        self.base_model_prefix_2 = ""

    def _convert_state_dict(self, flow_state_dict, cfg=None):
        """Convert state_dict's keys to match model.

        Args:
            flow_state_dict (OrderedDict): model state dict.
            cfg (dict): model's default config dict.

        Returns:
            OrderedDict: flow state dict.
        """
        # The converted checkpoint.
        oneflow_state_dict = flow_state_dict.copy()

        # Get configs
        num_heads = cfg.get("num_heads")
        hidden_size = cfg.get("embed_dim")
        head_size = int(hidden_size / num_heads)

        # prefix
        has_prefix = any(s.startswith(self.base_model_prefix_1) for s in oneflow_state_dict)

        index_idx = 3 if has_prefix else 2

        old_keys = oneflow_state_dict.keys()

        for key in list(old_keys):

            # Convert vit's embedding layers
            if "embeddings" in key:
                if "cls_token" in key:
                    new_key = "cls_token"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "position_embeddings" in key:
                    new_key = "pos_embed"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "patch_embeddings.projection" in key:
                    if "weight" in key:
                        new_key = "patch_embed.proj.weight"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    elif "bias" in key:
                        new_key = "patch_embed.proj.bias"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            # Convert vit's layernorm layers
            elif "layernorm_before" in key:
                index_block = key.split(".")[index_idx]
                if "weight" in key:
                    new_key = "blocks." + index_block + ".input_layernorm.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = "blocks." + index_block + ".input_layernorm.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            elif "layernorm_after" in key:
                index_block = key.split(".")[index_idx]
                if "weight" in key:
                    new_key = "blocks." + index_block + ".post_attention_layernorm.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = "blocks." + index_block + ".post_attention_layernorm.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            # Convert vit's attention layers
            elif "attention" in key:
                index_block = key.split(".")[index_idx]
                if "attention.attention" in key:
                    if (
                        "blocks." + index_block + ".self_attention.query_key_value.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    q_w = key
                    k_w = q_w.replace("query", "key")
                    v_w = q_w.replace("query", "value")
                    q_b = q_w.replace("weight", "bias")
                    k_b = k_w.replace("weight", "bias")
                    v_b = v_w.replace("weight", "bias")

                    qkv_w = flow.cat(
                        (
                            oneflow_state_dict.pop(q_w),
                            oneflow_state_dict.pop(k_w),
                            oneflow_state_dict.pop(v_w),
                        ),
                        dim=0,
                    )
                    qkv_b = flow.cat(
                        (
                            oneflow_state_dict.pop(q_b),
                            oneflow_state_dict.pop(k_b),
                            oneflow_state_dict.pop(v_b),
                        ),
                        dim=-1,
                    )

                    qkv_w = self._fix_qkv_ordering(qkv_w, head_size, num_heads)
                    qkv_b = self._fix_qkv_ordering(qkv_b, head_size, num_heads)

                    new_key = "blocks." + index_block + ".self_attention.query_key_value.weight"
                    oneflow_state_dict[new_key] = qkv_w

                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = qkv_b

                elif "output" in key:
                    if "dense" in key:
                        if "weight" in key:
                            new_key = "blocks." + index_block + ".self_attention.dense.weight"
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                        if "bias" in key:
                            new_key = "blocks." + index_block + ".self_attention.dense.bias"
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            elif "intermediate" in key:
                index_block = key.split(".")[index_idx]
                if "weight" in key:
                    if (
                        "blocks." + index_block + ".mlp.dense_h_to_4h.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = "blocks." + index_block + ".mlp.dense_h_to_4h.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)

            elif "output" in key:
                index_block = key.split(".")[index_idx]
                if "dense.weight" in key:
                    if (
                        "blocks." + index_block + ".mlp.dense_4h_to_h.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = w.replace("weight", "bias")
                    new_key = "blocks." + index_block + ".mlp.dense_4h_to_h.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)

            elif "layernorm" in key:
                if "weight" in key:
                    new_key = "norm.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = "norm.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            elif "classifier" in key:
                if "weight" in key:
                    new_key = "head.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = "head.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
            else:
                oneflow_state_dict[key] = oneflow_state_dict.pop(key)

        return oneflow_state_dict

    def _load_config_from_json(self, config_file):
        """load config from `config.json`, and update default config.

        Args:
            config_file (str): Path of config file.
        """
        with open(config_file, mode="r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        # update libai_cfg by config.json
        self._update_cfg("img_size", cfg_dict["image_size"])
        self._update_cfg("patch_size", cfg_dict["patch_size"])
        self._update_cfg("in_chans", cfg_dict["num_channels"])
        self._update_cfg("embed_dim", cfg_dict["hidden_size"])
        self._update_cfg("depth", cfg_dict["num_hidden_layers"])
        self._update_cfg("num_heads", cfg_dict["num_attention_heads"])
        self._update_cfg("attn_drop_rate", cfg_dict["attention_probs_dropout_prob"])
        self._update_cfg("drop_rate", cfg_dict["hidden_dropout_prob"])

        # update libai_cfg by kwargs
        for k, v in self.kwargs.items():
            self._update_cfg(k, v)

        self._update_cfg_log()


class ViTLoaderLiBai(ModelLoaderLiBai):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_2 = ""
