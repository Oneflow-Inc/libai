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

from libai.data.structures import DistTensorData, Instance
from libai.inference.basic import BasePipeline


class TextClassificationPipeline(BasePipeline):
    def __init__(
        self,
        config_file,
        data_parallel=None,
        tensor_parallel=None,
        pipeline_parallel=None,
        model_path=None,
        **kwargs,
    ):
        super().__init__(
            config_file, data_parallel, tensor_parallel, pipeline_parallel, model_path, **kwargs
        )

    def update_cfg(
        self,
        data_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=1,
    ):
        super().update_cfg(data_parallel, tensor_parallel, pipeline_parallel)
        self.cfg.model.cfg.bias_dropout_fusion = False
        assert "num_labels" in self.cfg.model.cfg, "The model's config must contain num_labels"
        if "label2id" not in self.cfg.model.cfg:
            label2id = {"Label_" + str(i): i for i in range(self.cfg.model.cfg.num_labels)}
            id2label = {ind: label for label, ind in label2id.items()}
            self.cfg.model.cfg["label2id"] = label2id
            self.cfg.model.cfg["id2label"] = id2label

    def _parse_parameters(self, **pipeline_parameters):
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {**pipeline_parameters}
        return preprocess_params, forward_params, postprocess_params

    def preprocess(
        self,
        inputs,
        pad: bool = False,
        **kwargs,
    ) -> dict:
        # tokenizer encoder
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
        self, model_outputs_dict, function_to_apply=None, return_all_scores=False, **kwargs
    ) -> dict:
        # prepare
        num_labels = self.cfg.model.cfg.num_labels
        if function_to_apply is not None:
            function_to_apply = function_to_apply.lower()
            assert function_to_apply in [
                "sigmoid",
                "softmax",
                "none",
            ], f"Unrecognized `function_to_apply` argument: {function_to_apply}"
        else:
            if num_labels == 1:
                function_to_apply = "sigmoid"
            elif num_labels > 1:
                function_to_apply = "softmax"

        # process, logits: [num_labels]
        logits = model_outputs_dict["logits"][0]

        if function_to_apply == "sigmoid":
            scores = flow.sigmoid(logits)
        elif function_to_apply == "softmax":
            scores = flow.softmax(logits)
        else:
            scores = logits
        scores = scores.detach().numpy()
        if return_all_scores:
            return [
                {"label": self.cfg.model.cfg.id2label[i], "score": score.item()}
                for i, score in enumerate(scores)
            ]
        else:
            return {
                "label": self.cfg.model.cfg.id2label[scores.argmax().item()],
                "score": scores.max().item(),
            }
