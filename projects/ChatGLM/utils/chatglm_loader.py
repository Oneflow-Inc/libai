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

from libai.models.utils.model_loader.base_loader import ModelLoaderHuggerFace, ModelLoaderLiBai


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


if __name__ == "__main__":
    from libai.config import LazyConfig, default_argument_parser

    config_file = "projects/ChatGLM/configs/chatglm_config.py"
    cfg = LazyConfig.load(config_file)
    args = default_argument_parser().parse_args()
    from libai.engine import default_setup

    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    model_loader = ChatGLMLoaderHuggerFace(
        cfg.model,
        cfg.model.cfg,
        cfg.model.cfg.pretrained_model_path,
    )
    model = model_loader.load()
