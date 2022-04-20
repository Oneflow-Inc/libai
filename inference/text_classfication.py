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

import sys

import numpy as np
import oneflow as flow

sys.path.append(".")  # noqa
from inference.basic import BasePipeline
from libai.data.structures import DistTensorData, Instance


class TextClassificationPipeline(BasePipeline):
    def __init__(self, config_file, **kwargs):
        super().__init__(config_file, **kwargs)

    def _parse_parameters(self, **pipeline_parameters):
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}
        return preprocess_params, forward_params, postprocess_params

    def preprocess(
        self,
        inputs,
        pad: bool = False,
        **kwargs,
    ) -> dict:
        # tokenizer encoder
        # inputs = self.tokenizer.tokenize(inputs)
        # input_ids = self.tokenizer.convert_tokens_to_ids(inputs)
        input_ids = flow.tensor(np.array(self.tokenizer.encode(inputs)))
        padding_mask = flow.tensor(np.ones(input_ids.shape))
        # set batch size = 1
        input_ids = input_ids.unsqueeze(0)
        padding_mask = padding_mask.unsqueeze(0)

        # to global tensor
        model_input = Instance(
            input_ids=DistTensorData(input_ids),
            attention_mask=DistTensorData(padding_mask),
        )
        mdoel_input_dict = {}
        for key, value in model_input.get_fields().items():
            value.to_global()
            mdoel_input_dict[key] = value.tensor
        return mdoel_input_dict

    def forward(self, mdoel_input_dict) -> dict:
        model_outputs_dict = self.model(**mdoel_input_dict)
        return model_outputs_dict

    def postprocess(
        self,
        model_input_dict,
        topk=5,
    ) -> dict:
        prediction_scores = model_input_dict["prediction_scores"]
        print(prediction_scores)
        # do something according to args


if __name__ == "__main__":
    model = TextClassificationPipeline("configs/bert_large_pretrain.py")
    a = model("dog" * 10)
