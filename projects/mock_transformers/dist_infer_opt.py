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
from omegaconf import DictConfig
from oneflow.utils.global_view import global_mode
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.opt import modeling_opt

from libai.layers import Linear
from libai.utils import distributed as dist

# ------replace attention to libai------
temp_class = modeling_opt.OPTAttention


class LiBaiOPTAttention(temp_class):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim = kwargs["embed_dim"]
        bias = kwargs["bias"]
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias, parallel="col", dtype=flow.float16)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias, parallel="col", dtype=flow.float16)
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, parallel="col", dtype=flow.float16)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, parallel="row", dtype=flow.float16)


modeling_opt.OPTAttention = LiBaiOPTAttention


# ----------replace Decoder to libai -----
temp_class = modeling_opt.OPTDecoderLayer


class LiBaiOPTDecoderLayer(temp_class):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = args[0]
        self.fc1 = Linear(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            parallel="col",
            dtype=flow.float16,
        )
        self.fc2 = Linear(
            config.ffn_dim,
            self.embed_dim,
            bias=config.enable_bias,
            parallel="row",
            dtype=flow.float16,
        )


modeling_opt.OPTDecoderLayer = LiBaiOPTDecoderLayer


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

    with global_mode(True, **placement_sbp_dict):
        # initial and load model
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=flow.float16)

    # set model to cuda
    dist.set_device_type("cuda")
    model._apply(dist.convert_to_distributed_default_setting)
    # initial tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)

    # get input_ids
    prompt = "Hello, I'm am conscious and"
    input_ids = tokenizer(prompt, return_tensors="np").input_ids
    input_ids = flow.from_numpy(input_ids)
    input_ids = input_ids.to_global(
        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        placement=dist.get_layer_placement(0),
    )

    with global_mode(True, **placement_sbp_dict):
        generated_ids = model.generate(input_ids, max_length=30)
    out_put_ids = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    if dist.is_main_process():
        print(out_put_ids)
