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


class GLMLoaderHuggerFace(ModelLoaderHuggerFace):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)

        """NOTE: base_model_prefix_1 is GLM's prefix in Transformers.
        base_model_prefix_2 is GLM's prefix in LiBai."""
        self.base_model_prefix_1 = "glm"
        self.base_model_prefix_2 = "glm"

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
        num_heads = cfg.get("num_attention_heads")
        hidden_size = cfg.get("hidden_size")
        head_size = int(hidden_size / num_heads)

        # prefix
        has_prefix = any(s.startswith(self.base_model_prefix_1) for s in oneflow_state_dict)
        prefix1 = self.base_model_prefix_1 + "." if has_prefix else ""
        prefix2 = "glm." if has_prefix else ""

        # Convert Embedding layers.
        new_key = prefix2 + "embeddings.word_embeddings.weight"
        old_keys.remove(prefix1 + "word_embeddings.weight")
        oneflow_state_dict[new_key] = oneflow_state_dict.pop(prefix1 + "word_embeddings.weight")

        if cfg.get("block_position_encoding", False) is True:
            new_key = prefix2 + "embeddings.position_embeddings.weight"
            old_keys.remove(prefix1 + "transformer.position_embeddings.weight")
            oneflow_state_dict[new_key] = oneflow_state_dict.pop(
                prefix1 + "transformer.position_embeddings.weight"
            )

            new_key = prefix2 + "embeddings.block_position_embeddings.weight"
            old_keys.remove(prefix1 + "transformer.block_position_embeddings.weight")
            oneflow_state_dict[new_key] = oneflow_state_dict.pop(
                prefix1 + "transformer.block_position_embeddings.weight"
            )

        # Convert other layers.
        for key in old_keys:
            if "query_key_value" in key:
                qkv = oneflow_state_dict.pop(key)
                qkv = self._fix_qkv_ordering(qkv, head_size, num_heads)
                oneflow_state_dict[prefix2 + key] = qkv
            else:
                oneflow_state_dict[prefix2 + key] = oneflow_state_dict.pop(key)

        return oneflow_state_dict

    def _load_config_from_json(self, config_file):
        """load config from `config.json`, and update default config.

        Args:
            config_file (str): Path of config file.
        """
        with open(config_file, mode="r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        # update libai_cfg by config.json
        for k, v in cfg_dict.items():
            self._update_cfg(k, v)

        # update libai_cfg by kwargs
        for k, v in self.kwargs.items():
            self._update_cfg(k, v)

        self._update_cfg_log()


class GLMLoaderLiBai(ModelLoaderLiBai):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_2 = "glm"

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
        num_heads = cfg.get("num_attention_heads")
        hidden_size = cfg.get("hidden_size")
        head_size = int(hidden_size / num_heads)

        # prefix
        has_prefix = any(s.startswith(self.base_model_prefix_1) for s in oneflow_state_dict)
        prefix1 = self.base_model_prefix_1 + "." if has_prefix else ""
        prefix2 = "glm." if has_prefix else ""

        # Convert Embedding layers.
        new_key = prefix2 + "embeddings.word_embeddings.weight"
        old_keys.remove(prefix1 + "word_embeddings.weight")
        oneflow_state_dict[new_key] = oneflow_state_dict.pop(prefix1 + "word_embeddings.weight")

        if cfg.get("block_position_encoding", False) is True:
            new_key = prefix2 + "embeddings.position_embeddings.weight"
            old_keys.remove(prefix1 + "transformer.position_embeddings.weight")
            oneflow_state_dict[new_key] = oneflow_state_dict.pop(
                prefix1 + "transformer.position_embeddings.weight"
            )

            new_key = prefix2 + "embeddings.block_position_embeddings.weight"
            old_keys.remove(prefix1 + "transformer.block_position_embeddings.weight")
            oneflow_state_dict[new_key] = oneflow_state_dict.pop(
                prefix1 + "transformer.block_position_embeddings.weight"
            )

        # Convert other layers.
        for key in old_keys:
            if "query_key_value" in key:
                qkv = oneflow_state_dict.pop(key)
                qkv = self._fix_qkv_ordering(qkv, head_size, num_heads)
                oneflow_state_dict[prefix2 + key] = qkv
            else:
                oneflow_state_dict[prefix2 + key] = oneflow_state_dict.pop(key)

        return oneflow_state_dict
