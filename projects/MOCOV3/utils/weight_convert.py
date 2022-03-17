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
from statistics import mode

import torch
import oneflow as flow

logger = logging.getLogger(__name__)

def filter_keys(key, value):
    """Filtering the state_dict keys and values to match LiBai's MAE model
    """
    if "norm1" in key:
        key = key.replace("norm1", "input_layernorm")
    elif "attn.qkv" in key:
        key = key.replace("attn.qkv", "self_attention.query_key_value")
        if "weight" in key:
            value = value.transpose((-1, -2))
    elif "attn.proj" in key:
        key = key.replace("attn.proj", "self_attention.dense")
        if "weight" in key:
            value = value.transpose((-1, -2))
    elif "norm2" in key:
        key = key.replace("norm2", "post_attention_layernorm")
    elif "mlp.fc1" in key:
        key = key.replace("mlp.fc1", "mlp.dense_h_to_4h")
        if "weight" in key:
            value = value.transpose((-1, -2))
    elif "mlp.fc2" in key:
        key = key.replace("mlp.fc2", "mlp.dense_4h_to_h")
        if "weight" in key:
            value = value.transpose((-1, -2))
    elif "head.weight" in key:
            value = value.transpose((-1, -2))
    elif "fc_norm" in key:
        key = key.replace("fc_norm", "norm")
    
    return key, value

def load_torch_checkpoint(model, path="./mae_finetuned_vit_base.pth", strict=False, linear_keyword="head"):
    """Load checkpoint from the given torch weights.
    Torch weight from: xxx
    """
    torch_dict = torch.load(path)["state_dict"]
    parameters = torch_dict
    new_parameters = dict()
    for key, value in parameters.items():
        if "num_batches_tracked" not in key:
            if key.startswith('module.base_encoder') and not key.startswith('module.base_encoder.%s' % linear_keyword):
                val = value.detach().cpu().numpy()
                # to global tensor
                key, val = filter_keys(key, val)
                val = flow.tensor(val).to_global(sbp=flow.sbp.broadcast, placement=flow.placement("cuda", {0: range(1)}))
                new_parameters[key[len("module.base_encoder."):]] = val

    model.load_state_dict(new_parameters, strict=strict)

    print("Successfully load torch mocov3 checkpoint.")
    return model


def load_torch_checkpoint_linear(model, path="./mae_finetuned_vit_base.pth", strict=False, linear_keyword="head"):
    """Load checkpoint from the given torch weights.
    Torch weight from: xxx
    """
    torch_dict = torch.load(path)["state_dict"]
    parameters = torch_dict
    new_parameters = dict()
    for key, value in parameters.items():
        if "num_batches_tracked" not in key:
            val = value.detach().cpu().numpy()
            # to global tensor
            key, val = filter_keys(key, val)
            val = flow.tensor(val).to_global(sbp=flow.sbp.broadcast, placement=flow.placement("cuda", {0: range(1)}))
            new_parameters[key[len("module."):]] = val

    model.load_state_dict(new_parameters, strict=strict)
    print("Successfully load torch mocov3 checkpoint.")
    return model
