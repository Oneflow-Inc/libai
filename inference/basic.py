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

from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import oneflow as flow

from libai.config import LazyConfig, try_get_key
from libai.engine.default import DefaultTrainer
from libai.utils import distributed as dist
from libai.utils.checkpoint import Checkpointer


class BasePipeline(metaclass=ABCMeta):
    """
    Base class for all task pipeline
    """

    def __init__(
        self,
        config_file,
        **kwargs,
    ):
        self.cfg = LazyConfig.load(config_file)
        dist.setup_dist_util(self.cfg.train.dist)
        self.model = DefaultTrainer.build_model(self.cfg).eval()
        self.load_pretrain_weight(self.model, self.cfg)
        self.tokenizer = self.build_tokenizer(self.cfg)
        (
            self._preprocess_params,
            self._forward_params,
            self._postprocess_params,
        ) = self._parse_parameters(
            **kwargs
        )  # noqa

    def load_pretrain_weight(self, model, cfg):
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=False
        )

    def build_tokenizer(self, cfg):
        tokenizer = None
        if try_get_key(cfg, "tokenization") is not None:
            tokenizer = DefaultTrainer.build_tokenizer(cfg)
        return tokenizer

    @abstractmethod
    def _parse_parameters(self, **pipeline_parameters):
        raise NotImplementedError("_parse_parameters not implemented")

    def __call__(self, inputs, *args, batch_size=None, **kwargs) -> dict:

        preprocess_params, forward_params, postprocess_params = self._parse_parameters(
            **kwargs
        )  # noqa

        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        with flow.no_grad():
            model_inputs_dict = self.preprocess(inputs, **preprocess_params)
            model_outputs_dict = self.forward(model_inputs_dict, **forward_params)
            model_outputs_dict = self.to_local(model_outputs_dict)
            outputs_dict = self.postprocess(model_outputs_dict, **postprocess_params)
        return outputs_dict

    def to_local(self, model_outputs_dict):
        model_outputs_dict = {
            key: dist.ttol(value, ranks=[0] if value.placement.ranks.ndim == 1 else [[0]])
            for key, value in model_outputs_dict.items()
        }

        if flow.cuda.is_available():
            dist.synchronize()
        return model_outputs_dict

    @abstractmethod
    def preprocess(self, input_: Any, **preprocess_parameters: Dict) -> dict:
        raise NotImplementedError("preprocess not implemented")

    @abstractmethod
    def forward(self, **kwargs: Dict) -> dict:
        raise NotImplementedError("forward not implemented")

    @abstractmethod
    def postprocess(self, **kwargs: Dict) -> dict:
        raise NotImplementedError("postprocess not implemented")
