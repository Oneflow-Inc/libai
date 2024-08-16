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

import init_env  # noqa
import oneflow as flow
from Baichuan import modeling_baichuan
from omegaconf import DictConfig
from oneflow.utils.global_view import global_mode
from transformers import AutoModelForCausalLM, AutoTokenizer

from libai.layers import Linear, RMSLayerNorm
from libai.utils import distributed as dist

# ------replace RMSNorm to libai------
modeling_baichuan.RMSNorm = RMSLayerNorm


# ----------replace MLP to libai -----
temp_class = modeling_baichuan.MLP


class LiBaiMLP(temp_class):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__(
            hidden_size,
            intermediate_size,
            hidden_act,
        )

        self.gate_proj = Linear(
            hidden_size, intermediate_size, bias=False, parallel="col", dtype=flow.float16
        )
        self.up_proj = Linear(
            hidden_size, intermediate_size, bias=False, parallel="col", dtype=flow.float16
        )
        self.down_proj = Linear(
            intermediate_size, hidden_size, bias=False, parallel="row", dtype=flow.float16
        )


modeling_baichuan.MLP = LiBaiMLP


# ----------replace Attention to libai -----
temp_class = modeling_baichuan.Attention


class LiBaiAttention(temp_class):
    def __init__(self, config):
        super().__init__(config)

        self.W_pack = Linear(
            self.hidden_size, 3 * self.hidden_size, bias=False, parallel="col", dtype=flow.float16
        )
        self.o_proj = Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            parallel="row",
            dtype=flow.float16,
        )


modeling_baichuan.Attention = LiBaiAttention


if __name__ == "__main__":
    # set dist config
    parallel_config = DictConfig(
        dict(
            data_parallel_size=1,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,  # set to 1, unsupport pipeline parallel now
            pipeline_num_layers=None,
            device_type="cpu",
        )
    )
    dist.setup_dist_util(parallel_config)

    placement_sbp_dict = dict(
        placement=flow.env.all_device_placement("cuda"),
        sbp=flow.sbp.broadcast,
    )

    # initial and load model
    with global_mode(True, **placement_sbp_dict):
        model = AutoModelForCausalLM.from_pretrained(
            "libai/projects/mock_transformers/Baichuan",
            torch_dtype=flow.float16,
            trust_remote_code=True,
        )

    # set model to cuda
    dist.set_device_type("cuda")
    model._apply(dist.convert_to_distributed_default_setting)
    # initial tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "libai/projects/mock_transformers/Baichuan", trust_remote_code=True
    )

    # get input_ids
    prompt = "登鹳雀楼->王之涣\n夜雨寄北->"
    input_ids = tokenizer(prompt, return_tensors="np").input_ids
    input_ids = flow.from_numpy(input_ids)
    input_ids = input_ids.to_global(
        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        placement=dist.get_layer_placement(0),
    )

    # generate id
    with global_mode(True, **placement_sbp_dict):
        generated_ids = model.generate(input_ids, max_new_tokens=64, repetition_penalty=1.1)
    out_put_ids = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    if dist.is_main_process():
        print(out_put_ids[0])
