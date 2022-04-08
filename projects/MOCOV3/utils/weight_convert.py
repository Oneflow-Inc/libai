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


import logging

import oneflow as flow
import torch

logger = logging.getLogger(__name__)


def convert_qkv_weight(value, num_heads, hidden_size):
    """
    convert qkv.weight to be compatible with LiBai transformer layer

    Args:
        cfg: config file
        value: qkv.weight in the loaded checkpoint
    """

    head_size = int(hidden_size / num_heads)
    qkv_weight = (
        value.view(3, num_heads, head_size, hidden_size)
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(hidden_size * 3, hidden_size)
    )

    return qkv_weight


def convert_qkv_bias(value, num_heads, hidden_size):
    """
    convert qkv.bias to be compatible with LiBai transformer layer

    Args:
        cfg: config file
        value: qkv.bias in the loaded checkpoint
    """

    head_size = int(hidden_size / num_heads)
    qkv_bias = (
        value.view(3, num_heads, head_size).permute(1, 0, 2).contiguous().view(hidden_size * 3)
    )

    return qkv_bias


def filter_keys(key, value, num_heads, hidden_size):
    """Filtering the state_dict keys and values to match LiBai's MOCOV3 model"""
    if "norm1" in key:
        key = key.replace("norm1", "input_layernorm")
    elif "attn.qkv" in key:
        key = key.replace("attn.qkv", "self_attention.query_key_value")
        if "weight" in key:
            value = convert_qkv_weight(value, num_heads, hidden_size)
        if "bias" in key:
            value = convert_qkv_bias(value, num_heads, hidden_size)
    elif "attn.proj" in key:
        key = key.replace("attn.proj", "self_attention.dense")
    elif "norm2" in key:
        key = key.replace("norm2", "post_attention_layernorm")
    elif "mlp.fc1" in key:
        key = key.replace("mlp.fc1", "mlp.dense_h_to_4h")
    elif "mlp.fc2" in key:
        key = key.replace("mlp.fc2", "mlp.dense_4h_to_h")
    elif "fc_norm" in key:
        key = key.replace("fc_norm", "norm")

    return key, value


def load_torch_checkpoint_linear_prob(
    num_heads, hidden_size, path="projects/MOCOV3/output/vit-b-300ep.pth.tar", linear_keyword="head"
):
    """Load checkpoint from the given torch weights.
    Torch weight from: xxx
    """
    torch_dict = torch.load(path, map_location="cpu")["state_dict"]
    parameters = torch_dict
    new_parameters = dict()
    for key, value in parameters.items():
        if "num_batches_tracked" not in key:
            if key.startswith("module.base_encoder") and not key.startswith(
                "module.base_encoder.%s" % linear_keyword
            ):
                # to global tensor
                key, val = filter_keys(key, value, num_heads, hidden_size)
                val = val.detach().cpu().numpy()
                val = flow.tensor(val).to_global(
                    sbp=flow.sbp.broadcast, placement=flow.placement("cuda", {0: range(1)})
                )
                new_parameters[key[len("module.base_encoder.") :]] = val
    return new_parameters
