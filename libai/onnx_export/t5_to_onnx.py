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
from projects.MT5.mt5_model import MT5Model
from projects.MT5.utils.mt5_loader import T5LoaderHuggerFace


def get_model(config_file):
    cfg = LazyConfig.load(config_file)

    cfg.model.cfg.model_type = "mt5"
    cfg.model.cfg.pretrained_model_path = None
    cfg.dataloader = None
    cfg.tokenization = None

    print("Building model....")
    loader = T5LoaderHuggerFace(MT5Model, cfg.model.cfg, "/path/to/model")
    model = loader.load()
    print("Build model finished.")

    return model


class t5Graph(nn.Graph):
    def __init__(self, eager_model):
        super().__init__()
        self.model = eager_model

    def build(
        self,
        encoder_input_ids,
        encoder_attn_mask,
        decoder_input_ids,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
    ):
        out = self.model(
            encoder_input_ids,
            encoder_attn_mask,
            decoder_input_ids,
            decoder_attn_mask,
            encoder_decoder_attn_mask,
        )
        return out


if __name__ == "__main__":
    model = get_model("projects/MT5/configs/mt5_pretrain.py")
    model.eval()

    t5_graph = t5Graph(model)
    # Build the static graph model
    encoder_input_ids = flow.ones(
        1, 5, dtype=flow.int64, sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0])
    )
    encoder_attn_mask = flow.ones(
        1, 3, dtype=flow.int64, sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0])
    )
    decoder_input_ids = flow.ones(
        1,
        5,
        5,
        dtype=flow.bool,
        sbp=flow.sbp.broadcast,
        placement=flow.placement("cuda", ranks=[0]),
    )
    decoder_attn_mask = flow.ones(
        1,
        3,
        3,
        dtype=flow.bool,
        sbp=flow.sbp.broadcast,
        placement=flow.placement("cuda", ranks=[0]),
    )
    encoder_decoder_attn_mask = flow.ones(
        1,
        3,
        5,
        dtype=flow.bool,
        sbp=flow.sbp.broadcast,
        placement=flow.placement("cuda", ranks=[0]),
    )

    # check your model.forward is valid
    # output = t5_graph(
    #     encoder_input_ids,
    #     encoder_attn_mask,
    #     decoder_input_ids,
    #     decoder_attn_mask,
    #     encoder_decoder_attn_mask
    # )
    # print(output)

    print("Compiling the graph which may make some time, please wait for a moment....")
    t5_graph._compile(
        encoder_input_ids,
        encoder_attn_mask,
        decoder_input_ids,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
    )

    convert_to_onnx_and_check(
        t5_graph,
        external_data=False,
        opset=11,
        flow_weight_dir=None,
        onnx_model_path="./",
        dynamic_batch_size=False,
        device="gpu_global",
        input_tensor_range=[0, 10],
    )
