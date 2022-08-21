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

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import oneflow as flow

from libai.config import LazyConfig, try_get_key
from libai.engine import DefaultTrainer
from libai.utils import distributed as dist
from libai.utils.checkpoint import Checkpointer
from libai.utils.logger import setup_logger

logger = setup_logger(distributed_rank=dist.get_rank())
logger = logging.getLogger("libai.inference")


class BasePipeline(metaclass=ABCMeta):
    """
    Base class for all task pipeline
    """

    def __init__(
        self,
        config_file,
        data_parallel=None,
        tensor_parallel=None,
        pipeline_parallel=None,
        pipeline_stage_id=None,
        model_path=None,
        **kwargs,
    ):
        # init cfg
        self.cfg = LazyConfig.load(config_file)
        flow.boxing.nccl.set_fusion_threshold_mbytes(
            try_get_key(self.cfg, "train.nccl_fusion_threshold_mb", default=16)
        )
        flow.boxing.nccl.set_fusion_max_ops_num(
            try_get_key(self.cfg, "train.nccl_fusion_max_ops", default=24)
        )
        self.update_cfg(
            data_parallel, 
            tensor_parallel, 
            pipeline_parallel,
            pipeline_stage_id,
        )
        dist.setup_dist_util(self.cfg.train.dist)
        assert (
            self.cfg.train.dist.data_parallel_size == 1
        ), "not support data parallel yet, only support tensor and pipeline parallel"
        logger.info(self.cfg.train.dist)

        # initial and load model
        self.model = self.load_pretrain_weight(self.cfg.model, model_path)
        self.model = self.model.eval()

        # initial tokenizer
        self.tokenizer = self.build_tokenizer(self.cfg)

        # set parameters
        (
            self._preprocess_params,
            self._forward_params,
            self._postprocess_params,
        ) = self._parse_parameters(**kwargs)

    def update_cfg(
        self,
        data_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=1,
        pipeline_stage_id=None,
    ):
        self.cfg.train.dist.data_parallel_size = data_parallel
        self.cfg.train.dist.tensor_parallel_size = tensor_parallel
        self.cfg.train.dist.pipeline_parallel_size = pipeline_parallel
        self.cfg.train.dist.custom_pipeline_stage_id = pipeline_stage_id
        if self.cfg.train.dist.pipeline_parallel_size > 1:
            assert (
                try_get_key(self.cfg.train.dist, "pipeline_num_layers") is not None
            ), "cfg.train.dist.pipeline_num_layers must be set when run pipeline parallel"

    def load_pretrain_weight(
            self, 
            libai_cfg_model, 
            model_path,
            base_model_prefix_1=None,
            base_model_prefix_2="",
            mode="libai"
        ):
        """load pretrained model.

        Args:
            libai_cfg_model (libai.models): Lazy config Model in Libai, you can import it 
                by `from libai.config.configs.common.models.bert 
                    import pretrain_model as libai_cfg_model` 
            model_path (str): The directory path of pretrained model,
        """
        print(mode)
        if mode == "libai":
            from libai.models.utils.model_utils.base_loader import ModelLoaderLiBai

            model_loader = ModelLoaderLiBai(
                libai_cfg_model, 
                libai_cfg_model.cfg,
                model_path
                )
            model_loader.base_model_prefix_1 = base_model_prefix_1
            model_loader.base_model_prefix_2 = base_model_prefix_2
            return model_loader.load()
        elif mode == "random":
            return DefaultTrainer.build_model(self.cfg)
        else:
            raise NotImplementedError

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
            if dist.is_main_process():
                outputs_dict = self.postprocess(model_outputs_dict, **postprocess_params)
            else:
                outputs_dict = {}
            dist.synchronize()
        return outputs_dict

    def to_local(self, model_outputs_dict):
        for key, value in model_outputs_dict.items():
            if isinstance(value, flow.Tensor) and value.is_global:
                model_outputs_dict[key] = dist.ttol(
                    value, ranks=[0] if value.placement.ranks.ndim == 1 else [[0]]
                )
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
