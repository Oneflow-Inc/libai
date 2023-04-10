# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
# Copyright 2022 The HuggingFace Inc. team.
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

from libai.utils import distributed as dist
from projects.GLM.configs.glm_inference import cfg
from projects.GLM.modeling_glm import GLMForConditionalGeneration
from projects.GLM.tokenizer.glm_tokenizer import GLMChineseTokenzier
from projects.GLM.utils.glm_loader import GLMLoaderHuggerFace

tokenizer = GLMChineseTokenzier.from_pretrained("/data/home/xiezipeng/glm-10b-chinese")
input_ids = tokenizer.encode(
    ["西游记的作者是[MASK]。"],
    return_tensors="of",
)
inputs = {"input_ids": input_ids, "attention_mask": flow.ones(input_ids.size(), dtype=flow.bool)}
inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)

sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
placement = dist.get_layer_placement(0)

dist.set_device_type("cpu")
loader = GLMLoaderHuggerFace(
    GLMForConditionalGeneration,
    cfg,
    "/data/home/xiezipeng/glm-10b-chinese",
    embedding_dropout_prob=0,
    attention_dropout_prob=0,
    output_dropout_prob=0,
)
model = loader.load()
model = model.half().cuda()
model.eval()

dist.set_device_type("cuda")

while True:
    outputs = model.generate(
        inputs=inputs["input_ids"].to_global(sbp=sbp, placement=placement),
        position_ids=inputs["position_ids"].to_global(sbp=sbp, placement=placement),
        generation_attention_mask=inputs["generation_attention_mask"].to_global(
            sbp=sbp, placement=placement
        ),
        max_length=512,
    )

    res = tokenizer.decode(outputs[0])
    if dist.is_main_process():
        print(res)
