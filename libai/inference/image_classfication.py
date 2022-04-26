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

import os

import oneflow as flow
from PIL import Image

from libai.config import instantiate
from libai.data.structures import DistTensorData, Instance
from libai.inference.basic import BasePipeline


class ImageClassificationPipeline(BasePipeline):
    def __init__(self, config_file, **kwargs):
        super().__init__(config_file, **kwargs)
        assert "num_classes" in self.cfg.model, "The model's config must contain num_classes"
        self.label2id = {"Label_" + str(i): i for i in range(self.cfg.model.num_classes)}
        self.id2label = {ind: label for label, ind in self.label2id.items()}
        self.transform = instantiate(self.cfg.dataloader.test[0].dataset.transform)

    def _parse_parameters(self, **pipeline_parameters):
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {**pipeline_parameters}
        return preprocess_params, forward_params, postprocess_params

    def preprocess(
        self,
        inputs,
        **kwargs,
    ) -> dict:
        assert os.path.exists(inputs), "inputs must be an existing image path!"
        with open(inputs, "rb") as f:
            img = Image.open(f).convert("RGB")
        img = self.transform(img)
        img = img.unsqueeze(0)

        # to global tensor
        model_input = Instance(
            images=DistTensorData(img),
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
        num_labels = self.cfg.model.num_classes
        if function_to_apply is not None:
            function_to_apply = function_to_apply.lower()
            assert function_to_apply in [
                "sigmoid",
                "softmax",
                "none",
            ], f"Unrecognized `function_to_apply` argument: {function_to_apply}"
        else:
            if num_labels == 1:
                function_to_apply == "sigmoid"
            elif num_labels > 1:
                function_to_apply == "softmax"

        # process, logits: [num_labels]
        logits = model_outputs_dict["prediction_scores"][0]

        if function_to_apply == "sigmoid":
            scores = flow.sigmoid(logits)
        elif function_to_apply == "softmax":
            scores = flow.softmax(logits)
        else:
            scores = logits
        scores = scores.detach().numpy()

        if return_all_scores:
            return [
                {"label": self.id2label[i], "score": score.item()} for i, score in enumerate(scores)
            ]
        else:
            return {
                "label": self.id2label[scores.argmax().item()],
                "score": scores.max().item(),
            }


if __name__ == "__main__":
    for i in range(100):
        model = ImageClassificationPipeline("configs/swin_imagenet.py")
        a = model("/DATA/disk1/ImageNet/extract/val/n01440764/ILSVRC2012_val_00000293.JPEG")
        b = model("/DATA/disk1/ImageNet/extract/val/n01440764/ILSVRC2012_val_00000293.JPEG")
        print(a, b)
