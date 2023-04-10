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


class SwinLoaderHuggerFace(ModelLoaderHuggerFace):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)

        """NOTE: base_model_prefix_1 is SWIN's prefix in Transformers.
        base_model_prefix_2 is SWIN's prefix in LiBai."""

        self.base_model_prefix_1 = "swin"
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

        # prefix
        has_prefix = any(s.startswith(self.base_model_prefix_1) for s in oneflow_state_dict)

        index_idx_1 = 3 if has_prefix else 2
        index_idx_2 = 5 if has_prefix else 4

        old_keys = oneflow_state_dict.keys()

        for key in list(old_keys):

            # Convert swin's embedding layers
            if "embeddings" in key:
                if "patch_embeddings.projection" in key:
                    if "weight" in key:
                        new_key = "patch_embed.proj.weight"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    elif "bias" in key:
                        new_key = "patch_embed.proj.bias"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "norm" in key:
                    if "weight" in key:
                        new_key = "patch_embed.norm.weight"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    elif "bias" in key:
                        new_key = "patch_embed.norm.bias"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            # Convert swin's layernorm layers
            elif "layernorm_before" in key:
                index_layer = key.split(".")[index_idx_1]
                index_block = key.split(".")[index_idx_2]
                if "weight" in key:
                    new_key = "layers." + index_layer + ".blocks." + index_block + ".norm1.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = "layers." + index_layer + ".blocks." + index_block + ".norm1.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            elif "layernorm_after" in key:
                index_layer = key.split(".")[index_idx_1]
                index_block = key.split(".")[index_idx_2]
                if "weight" in key:
                    new_key = "layers." + index_layer + ".blocks." + index_block + ".norm2.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = "layers." + index_layer + ".blocks." + index_block + ".norm2.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            # Convert swin's attention layers
            elif "attention" in key:
                index_layer = key.split(".")[index_idx_1]
                index_block = key.split(".")[index_idx_2]
                if "self" in key:
                    if (
                        "relative_position_bias_table" in key
                    ):  # convert relative_position_bias_table but not index
                        new_key = (
                            "layers."
                            + index_layer
                            + ".blocks."
                            + index_block
                            + ".attn.relative_position_bias_table"
                        )
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    elif "relative_position_index" in key:
                        new_key = (
                            "layers."
                            + index_layer
                            + ".blocks."
                            + index_block
                            + ".attn.relative_position_index"
                        )
                        oneflow_state_dict.pop(key)
                    else:
                        if (
                            "layers." + index_layer + ".blocks." + index_block + ".attn.qkv.weight"
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

                        new_key = (
                            "layers." + index_layer + ".blocks." + index_block + ".attn.qkv.weight"
                        )
                        oneflow_state_dict[new_key] = qkv_w

                        new_key = new_key.replace("weight", "bias")
                        oneflow_state_dict[new_key] = qkv_b

                elif "output" in key:
                    if "dense" in key:
                        if "weight" in key:
                            new_key = (
                                "layers."
                                + index_layer
                                + ".blocks."
                                + index_block
                                + ".attn.proj.weight"
                            )
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                        if "bias" in key:
                            new_key = (
                                "layers."
                                + index_layer
                                + ".blocks."
                                + index_block
                                + ".attn.proj.bias"
                            )
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            elif "intermediate" in key:
                index_layer = key.split(".")[index_idx_1]
                index_block = key.split(".")[index_idx_2]
                if "weight" in key:
                    if (
                        "layers."
                        + index_layer
                        + ".blocks."
                        + index_block
                        + ".mlp.dense_h_to_4h.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = (
                        "layers."
                        + index_layer
                        + ".blocks."
                        + index_block
                        + ".mlp.dense_h_to_4h.weight"
                    )
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)

            elif "output" in key:
                index_layer = key.split(".")[index_idx_1]
                index_block = key.split(".")[index_idx_2]
                if "dense.weight" in key:
                    if (
                        "layers."
                        + index_layer
                        + ".blocks."
                        + index_block
                        + ".mlp.dense_4h_to_h.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = w.replace("weight", "bias")
                    new_key = (
                        "layers."
                        + index_layer
                        + ".blocks."
                        + index_block
                        + ".mlp.dense_4h_to_h.weight"
                    )
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)

            elif "downsample" in key:
                index_layer = key.split(".")[index_idx_1]
                if "reduction.weight" in key:
                    new_key = "layers." + index_layer + ".downsample.reduction.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "norm" in key:
                    if (
                        "layers." + index_layer + ".downsample.norm.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = w.replace("weight", "bias")
                    new_key = "layers." + index_layer + ".downsample.norm.weight"
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
        self._update_cfg("embed_dim", cfg_dict["embed_dim"])
        self._update_cfg("depths", cfg_dict["depths"])
        self._update_cfg("num_heads", cfg_dict["num_heads"])
        self._update_cfg("window_size", cfg_dict["window_size"])
        self._update_cfg("mlp_ratio", cfg_dict["mlp_ratio"])
        self._update_cfg("qkv_bias", cfg_dict["qkv_bias"])
        self._update_cfg("drop_path_rate", cfg_dict["drop_path_rate"])

        # update libai_cfg by kwargs
        for k, v in self.kwargs.items():
            self._update_cfg(k, v)

        self._update_cfg_log()


class SwinLoaderLiBai(ModelLoaderLiBai):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_2 = ""
