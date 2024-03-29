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

from libai.models.utils.model_loader.base_loader import ModelLoaderHuggerFace, ModelLoaderLiBai


class Qwen2LoaderHuggerFace(ModelLoaderHuggerFace):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)

        self.base_model_prefix_1 = "model"
        self.base_model_prefix_2 = "model"

    def _convert_state_dict(self, flow_state_dict, cfg):
        """Convert state_dict's keys to match model.

        Args:
            flow_state_dict (OrderedDict): model state dict.
            cfg (dict): model's default config dict in LiBai.

        Returns:
            OrderedDict: flow state dict.
        """
        # The converted checkpoint.
        oneflow_state_dict = flow_state_dict.copy()
        old_keys = list(oneflow_state_dict.keys())

        # Get configs
        num_attention_heads = cfg.get("num_attention_heads")
        hidden_size = cfg.get("hidden_size")
        head_size = int(hidden_size // num_attention_heads)

        new_key_qkv_w = "model.layers.{}.self_attn.query_key_value.weight"
        old_key_qkv_w = "model.layers.{}.self_attn.{}.weight"
        new_key_qkv_b = "model.layers.{}.self_attn.query_key_value.bias"
        old_key_qkv_b = "model.layers.{}.self_attn.{}.bias"
        for layer_idx in range(cfg.get("hidden_layers")):
            query_w = old_key_qkv_w.format(layer_idx, "q_proj")
            key_w = old_key_qkv_w.format(layer_idx, "k_proj")
            value_w = old_key_qkv_w.format(layer_idx, "v_proj")
            qw = oneflow_state_dict[query_w]
            kw = oneflow_state_dict[key_w]
            vw = oneflow_state_dict[value_w]
            qkv_w = flow.cat([qw, kw, vw], dim=0)
            qkv_w = self._fix_qkv_ordering(qkv_w, head_size, num_attention_heads, hidden_size)
            oneflow_state_dict[new_key_qkv_w.format(layer_idx)] = qkv_w
            oneflow_state_dict.pop(query_w)
            oneflow_state_dict.pop(key_w)
            oneflow_state_dict.pop(value_w)
            
            query_b = old_key_qkv_b.format(layer_idx, "q_proj")
            key_b = old_key_qkv_b.format(layer_idx, "k_proj")
            value_b = old_key_qkv_b.format(layer_idx, "v_proj")
            qb = oneflow_state_dict[query_b]
            kb = oneflow_state_dict[key_b]
            vb = oneflow_state_dict[value_b]
            qkv_b = flow.cat([qb, kb, vb], dim=0)
            qkv_b = self._fix_qkv_ordering(qkv_b, head_size, num_attention_heads, hidden_size)
            oneflow_state_dict[new_key_qkv_b.format(layer_idx)] = qkv_b
            oneflow_state_dict.pop(query_b)
            oneflow_state_dict.pop(key_b)
            oneflow_state_dict.pop(value_b)

        for k in old_keys:
            if "inv_freq" in k:
                oneflow_state_dict.pop(k)

        return oneflow_state_dict

    def _load_config_from_json(self, config_file):
        """load config from `config.json`, and update default config.

        Args:
            config_file (str): Path of config file.
        """
        with open(config_file, mode="r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        # update libai_cfg by config.json
        self._update_cfg("hidden_layers", cfg_dict["num_hidden_layers"])
        self._update_cfg("hidden_size", cfg_dict["hidden_size"])
        self._update_cfg("num_attention_heads", cfg_dict["num_attention_heads"])
        self._update_cfg("max_position_embeddings", cfg_dict["max_position_embeddings"])
        self._update_cfg("intermediate_size", cfg_dict["intermediate_size"])
        self._update_cfg("rms_norm_eps", cfg_dict["rms_norm_eps"])
        self._update_cfg("vocab_size", cfg_dict["vocab_size"])
        self._update_cfg("initializer_range", cfg_dict["initializer_range"])
        self._update_cfg(
            "ffn_hidden_size",
            cfg_dict.get("n_inner")
            if cfg_dict.get("n_inner") is not None
            else 4 * self.libai_cfg["hidden_size"],
        )

        # update libai_cfg by kwargs
        for k, v in self.kwargs.items():
            self._update_cfg(k, v)

        self._update_cfg_log()


class Qwen2LoaderLiBai(ModelLoaderLiBai):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_2 = "model"
