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

import numpy as np
import oneflow as flow
import torch

logger = logging.getLogger(__name__)


def load_torch_checkpoint(model, cfg, path="path/to/pytorch/model.pth", strict=False):
    """
    Load checkpoint from the given torch weights.
    Torch weight can be downloaded from the original repo:
        https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v4
    """
    # torch_dict = torch.load(path, map_location="cpu")["model"]
    torch_dict = torch.load(path, map_location="cpu")

    parameters = torch_dict
    new_parameters = dict()
    for key, value in parameters.items():
        if "num_batches_tracked" not in key:
            # to global tensor
            val = value.detach().float().cpu().numpy()
            val = flow.tensor(val).to_global(
                sbp=flow.sbp.broadcast, placement=flow.placement("cpu", ranks=[0])
            )
            new_parameters[key] = val
    model.load_state_dict(new_parameters, strict=strict)
    print("Successfully load torch RWKV checkpoint.")
    return model
