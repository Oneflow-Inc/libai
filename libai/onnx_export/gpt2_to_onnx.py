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


import oneflow as flow
from oneflow import nn
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check

from libai.config import LazyConfig
from libai.engine import DefaultTrainer


def get_model(config_file):
    cfg = LazyConfig.load(config_file)

    cfg.model.cfg.pretrained_model_path = None
    cfg.dataloader = None
    cfg.tokenization = None

    print("Building model....")
    model = DefaultTrainer.build_model(cfg)
    print("Build model finished.")

    return model

class gpt2Graph(nn.Graph):
    def __init__(self, eager_model):
        super().__init__()
        self.model = eager_model

    def build(
        self,
        input_ids,
    ):
        out = self.model(
            input_ids,
        )
        return out["prediction_scores"]


if __name__ == "__main__":
    model = get_model("projects/MagicPrompt/configs/gpt2_training.py")
    model.eval()

    gpt2_graph = gpt2Graph(model)
    # Build the static graph model
    input_ids = flow.ones(1, 5, dtype=flow.int64, sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0]))
    
    # check your model.forward is valid
    # output = gpt2_graph(
    #     input_ids
    # )

    print("Compiling the graph which may make some time, please wait for a moment....")

    gpt2_graph._compile(
        input_ids,
    )

    convert_to_onnx_and_check(
        gpt2_graph,
        external_data=False,
        opset=11,
        flow_weight_dir=None,
        onnx_model_path="./",
        dynamic_batch_size=False,
        device="gpu_global",
        input_tensor_range=[0, 10],
    )
