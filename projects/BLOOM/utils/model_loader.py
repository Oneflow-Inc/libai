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

from libai.models.utils import ModelLoaderHuggerFace, ModelLoaderLiBai


class BlooMLoaderHuggerFace(ModelLoaderHuggerFace):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)

        """NOTE: base_model_prefix_1 is BLOOM's prefix in Transformers.
        base_model_prefix_2 is BLOOM's prefix in LiBai."""
        self.base_model_prefix_1 = "transformer"
        self.base_model_prefix_2 = "transformer"

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

        # prefix
        has_prefix = any(s.startswith(self.base_model_prefix_1) for s in oneflow_state_dict)
        prefix2 = "transformer." if has_prefix else ""

        # Convert layers.
        for key in old_keys:
            oneflow_state_dict[prefix2 + key] = oneflow_state_dict.pop(key)

        return oneflow_state_dict

    def _load_config_from_json(self, config_file):
        """load config from `config.json`, and update default config.

        Args:
            config_file (str): Path of config file.
        """
        with open(config_file, mode="r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        self._update_cfg("hidden_layers", cfg_dict["n_layer"])
        self._update_cfg("hidden_size", cfg_dict["n_embed"])
        self._update_cfg("n_head", cfg_dict["num_attention_heads"])

        # update libai_cfg by config.json
        for k, v in cfg_dict.items():
            self._update_cfg(k, v)

        # update libai_cfg by kwargs
        for k, v in self.kwargs.items():
            self._update_cfg(k, v)

        self._update_cfg_log()


class BlooMLoaderLibai(ModelLoaderLiBai):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_2 = "transformer"
