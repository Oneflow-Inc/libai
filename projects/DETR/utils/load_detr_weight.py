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

from collections import OrderedDict

import oneflow as flow
import torch

import libai.utils.distributed as dist


def convert_tensor(tensor):
    tensor = tensor.float()
    return flow.Tensor(tensor.cpu().numpy())


def convert_state(state):
    save = OrderedDict()
    for name, tensor in state.items():
        if "in_proj" in name:
            if "weight" in name:
                save[name[:-14]+"query_key_value.weight"] = convert_tensor(tensor)
            elif "bias" in name:
                save[name[:-12]+"query_key_value.bias"] = convert_tensor(tensor)
        elif "out_proj" in name:
            if "weight" in name:
                save[name[:-15]+"dense.weight"] = convert_tensor(tensor)
            elif "bias" in name:
                save[name[:-13]+"dense.bias"] = convert_tensor(tensor)
        else:
            save[name] = convert_tensor(tensor)
    return save


def load_tensor(tensor_lhs, tensor_rhs):
    tensor_rhs = flow.to_global(
        tensor_rhs,
        placement=tensor_lhs.placement,
        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
    )
    tensor_rhs = tensor_rhs.to_global(sbp=tensor_lhs.sbp)
    tensor_lhs.copy_(tensor_rhs)


def load_detr_weights(model, path, hidden_size, num_heads, layers=12):
    head_size = hidden_size // num_heads
    detr_state_dict = torch.load(path)
    of_state_dict, _ = convert_state(
        detr_state_dict,
        layers=layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_size=head_size,
    )
    for key, value in of_state_dict.items():
        load_tensor(model.state_dict()[key], value)