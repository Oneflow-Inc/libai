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
from projects.ChatGLM.lora.lora_model import LoraModel


class ChatGLMLoaderHuggerFace(ModelLoaderHuggerFace):
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
        for key in cfg_dict:
            self._update_cfg(key, cfg_dict[key])

        # update libai_cfg by kwargs
        for k, v in self.kwargs.items():
            self._update_cfg(k, v)

        self._update_cfg_log()


class ChatGLMLoaderLiBai(ModelLoaderLiBai):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_2 = "model"


class ChatGLMLoraLoaderHuggerFace(ChatGLMLoaderHuggerFace):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.lora_cfg = kwargs.pop("lora_cfg")
        self.lora_pretrained_model_path = kwargs.pop("lora_pretrained_model_path")

    def load(self):
        if self.output_loading_info:
            model, loading_info_base = super().load()
            model, loading_info_lora = self._convert_to_lora_model(model)
            loading_info = dict()
            for key in loading_info_base:
                loading_info[key] = loading_info_base[key] + loading_info_lora[key]
            return model, loading_info
        model = super().load()
        model, _ = self._convert_to_lora_model(model)

        return model

    def _convert_to_lora_model(self, model):
        self.model = LoraModel(model, self.lora_cfg, adapter_name="default")
        loading_info = {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        }
        if self.lora_pretrained_model_path is not None:
            flow_state_dict = flow.load(self.lora_pretrained_model_path, global_src_rank=0)

            # State_dict to global
            self._state_dict_to_global(flow_state_dict, mode="libai")

            # Load
            (
                self.model,
                missing_keys,
                unexpected_keys,
                mismatched_keys,
                error_msgs,
            ) = self._load_pretrained_model(
                self.model, flow_state_dict, self.lora_pretrained_model_path
            )

            missing_keys = [key for key in missing_keys if "lora" in key.lower()]
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            if self.lora_cfg.inference_mode:
                self.model.merge_adapter()
        return self.model, loading_info


class ChatGLMLoraLoaderLiBai(ChatGLMLoaderLiBai):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.lora_cfg = kwargs.pop("lora_cfg")
        self.lora_pretrained_model_path = kwargs.pop("lora_pretrained_model_path")

    def load(self):
        if self.output_loading_info:
            model, loading_info_base = super().load()
            model, loading_info_lora = self._convert_to_lora_model(model)
            loading_info = dict()
            for key in loading_info_base:
                loading_info[key] = loading_info_base[key] + loading_info_lora[key]
            return model, loading_info
        model = super().load()
        model, _ = self._convert_to_lora_model(model)

        return model

    def _convert_to_lora_model(self, model):
        self.model = LoraModel(model, self.lora_cfg, adapter_name="default")
        loading_info = {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        }
        if self.lora_pretrained_model_path is not None:

            flow_state_dict = flow.load(self.lora_pretrained_model_path, global_src_rank=0)

            # State_dict to global
            self._state_dict_to_global(flow_state_dict, mode="libai")

            # Load
            (
                self.model,
                missing_keys,
                unexpected_keys,
                mismatched_keys,
                error_msgs,
            ) = self._load_pretrained_model(
                self.model, flow_state_dict, self.lora_pretrained_model_path
            )

            missing_keys = [key for key in missing_keys if "lora" in key.lower()]
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }

            if self.lora_cfg.inference_mode:
                self.model.merge_adapter()
        return self.model, loading_info


if __name__ == "__main__":
    from libai.config import LazyConfig, default_argument_parser

    config_file = "projects/ChatGLM/configs/chatglm_config.py"
    cfg = LazyConfig.load(config_file)
    args = default_argument_parser().parse_args()
    from libai.engine import default_setup

    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    model_loader = ChatGLMLoraLoaderHuggerFace(
        cfg.model,
        cfg.model.cfg,
        cfg.model.cfg.pretrained_model_path,
        lora_cfg=cfg.model.cfg.lora_cfg,
        lora_pretrained_model_path=cfg.model.cfg.lora_pretrained_model_path,
    )
    model = model_loader.load()
    print()
