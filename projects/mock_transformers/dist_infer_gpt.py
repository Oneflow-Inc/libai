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
from transformers import AutoModelForCausalLM, AutoTokenizer, pytorch_utils
from transformers.models.gpt2 import modeling_gpt2

from libai.layers import Conv1D
from libai.utils import distributed as dist


# ------replace Conv1D to libai------
class LiBaiConv1d(Conv1D):
    def __init__(
        self,
        nf,
        nx,
        bias=True,
        parallel="data",
        init_method=flow.nn.init.xavier_normal_,
        skip_bias_add=False,
        dtype=flow.float32,
        layer_idx=0,
    ):
        super().__init__(
            in_features=nx,
            out_features=nf,
            bias=bias,
            parallel=parallel,
            init_method=init_method,
            skip_bias_add=skip_bias_add,
            dtype=dtype,
            layer_idx=layer_idx,
        )


pytorch_utils.Conv1D = LiBaiConv1d


# ------replace attention to libai------
temp_class = modeling_gpt2.GPT2Attention


class LiBaiGPT2Attention(temp_class):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)

        if is_cross_attention:
            self.c_attn = Conv1D(
                in_features=self.embed_dim,
                out_features=2 * self.embed_dim,
                parallel="col",
                dtype=flow.float16,
            )
            self.q_attn = Conv1D(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                parallel="col",
                dtype=flow.float16,
            )
        else:
            self.c_attn = Conv1D(
                in_features=self.embed_dim,
                out_features=3 * self.embed_dim,
                parallel="col",
                dtype=flow.float16,
            )
        self.c_proj = Conv1D(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            parallel="row",
            dtype=flow.float16,
        )


modeling_gpt2.GPT2Attention = LiBaiGPT2Attention


# ------replace mlp to libai------
temp_class = modeling_gpt2.GPT2MLP


class LiBaiGPT2MLP(temp_class):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(
            in_features=embed_dim,
            out_features=intermediate_size,
            parallel="col",
            dtype=flow.float16,
        )
        self.c_proj = Conv1D(
            in_features=intermediate_size,
            out_features=embed_dim,
            parallel="row",
            dtype=flow.float16,
        )


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
        model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=flow.float16)

    # set model to cuda
    dist.set_device_type("cuda")
    model._apply(dist.convert_to_distributed_default_setting)
    # initial tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)

    # get input_ids
    prompt = "Hello, I'm a language model,"
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
