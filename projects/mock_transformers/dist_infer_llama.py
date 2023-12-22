# coding=utf-8
# Copyright 2021 The Sugon Authors. All rights reserved.
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
from omegaconf import DictConfig
from oneflow.utils.global_view import global_mode
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import modeling_llama

from libai.layers import Linear
from libai.utils import distributed as dist

# ------replace attention to libai------
temp_class = modeling_llama.LlamaAttention


class LiBaiLlamaAttention(temp_class):
    def __init__(self, config):
        super().__init__(config)
        self.q_proj = Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            parallel="col",
            dtype=flow.float16,
        )
        self.k_proj = Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            parallel="col",
            dtype=flow.float16,
        )
        self.v_proj = Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            parallel="col",
            dtype=flow.float16,
        )
        self.o_proj = Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            parallel="row",
            dtype=flow.float16,
        )


modeling_llama.LlamaAttention = LiBaiLlamaAttention

# ----------replace mlp to libai -----
temp_class = modeling_llama.LlamaMLP


class LiBaiLlamaMLP(temp_class):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.gate_proj = Linear(
            hidden_size, intermediate_size, bias=False, parallel="col", dtype=flow.float16
        )
        self.down_proj = Linear(
            intermediate_size, hidden_size, bias=False, parallel="col", dtype=flow.float16
        )
        self.up_proj = Linear(
            hidden_size, intermediate_size, bias=False, parallel="row", dtype=flow.float16
        )


modeling_llama.LlamaMLP = LiBaiLlamaMLP

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
            "meta-llama/Llama-2-7b", torch_dtype=flow.float16
        )

    # set model to cuda
    dist.set_device_type("cuda")
    model._apply(dist.convert_to_distributed_default_setting)
    # initial tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b", use_fast=False)

    # get input_ids
    prompt = "Hello, I'm am conscious and"
    input_ids = tokenizer(prompt, return_tensors="np").input_ids
    input_ids = flow.from_numpy(input_ids)
    input_ids = input_ids.to_global(
        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        placement=dist.get_layer_placement(0),
    )

    # generate id
    with global_mode(True, **placement_sbp_dict):
        generated_ids = model.generate(input_ids, max_length=30)
    out_put_ids = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    if dist.is_main_process():
        print(out_put_ids)
