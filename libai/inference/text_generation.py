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

import numpy as np
import oneflow as flow
from omegaconf import OmegaConf

from libai.config import LazyCall
from libai.data.structures import DistTensorData, Instance
from libai.inference.basic import BasePipeline
from libai.tokenizer import T5Tokenizer


class TextGenerationPipeline(BasePipeline):
    def __init__(
        self,
        config_file,
        data_parallel=None,
        tensor_parallel=None,
        pipeline_parallel=None,
        pipeline_stage_id=None,
        pipeline_num_layers=None,
        model_path=None,
        mode="libai",
        **kwargs,
    ):
        super().__init__(
            config_file,
            data_parallel,
            tensor_parallel,
            pipeline_parallel,
            pipeline_stage_id,
            pipeline_num_layers,
            model_path,
            mode,
            **kwargs,
        )

    def update_cfg(
        self,
        data_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=1,
        pipeline_stage_id=None,
        pipeline_num_layers=None,
    ):
        super().update_cfg(
            data_parallel,
            tensor_parallel,
            pipeline_parallel,
            pipeline_stage_id,
            pipeline_num_layers,
        )
        self.cfg.model.cfg.model_type = "t5"
        self.cfg.model.cfg.pretrained_model_path = None
        self.cfg.dataloader = None
        self.cfg.tokenization = OmegaConf.create()
        self.cfg.tokenization.append_eod = False
        self.cfg.tokenization.make_vocab_size_divisible_by = 128
        self.cfg.tokenization.tokenizer = LazyCall(T5Tokenizer)(
            vocab_file="data_test/t5_inference_model/spiece.model",
        )

    def load_pretrain_weight(self, libai_cfg_model, model_path, mode="huggingface"):
        """load pretrained model.

        Args:
            libai_cfg_model (libai.models): Lazy config Model in Libai, you can import it
                by `from libai.config.configs.common.models.bert
                    import pretrain_model as libai_cfg_model`
            model_path (str): The directory path of pretrained model,
        """
        if mode == "huggingface":
            from projects.MT5.utils.mt5_loader import T5LoaderHuggerFace

            model_loader = T5LoaderHuggerFace(
                libai_cfg_model,
                libai_cfg_model.cfg,
                model_path,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                embedding_dropout_prob=0.0,
                mlp_type="t5",
            )
            return model_loader.load()
        else:
            return super().load_pretrain_weight(
                libai_cfg_model,
                model_path,
                mode=mode,
            )

    def _parse_parameters(self, **pipeline_parameters):
        preprocess_params = {}
        forward_params = {**pipeline_parameters}
        postprocess_params = {}

        return preprocess_params, forward_params, postprocess_params

    def preprocess(
        self,
        inputs,
        pad: bool = False,
        **kwargs,
    ) -> dict:
        # tokenizer encoder
        encoder_ids = self.tokenizer.encode(
            inputs, 
            return_tensors="of",
            is_global=True
        )

        encoder_input_dict = {
            "encoder_ids": encoder_ids,
        }

        return encoder_input_dict 

    def forward(self, encoder_input_dict, **kwargs) -> dict:
        outputs = self.model.generate(
            encoder_input_dict["encoder_ids"],
            **kwargs
        )
        return {"return_ids": outputs}

    def postprocess(self, model_output_dict, **kwargs) -> dict:
        text = self.tokenizer.decode(
            model_output_dict["return_ids"][0], 
            skip_special_tokens=True
        )
        records = {"generated_text": text}
        return records
