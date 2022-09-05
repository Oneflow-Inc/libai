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


def get_RWKV_v4_config_optim(model, grad_clip):
    no_decay = set()

    for mn, m in model.named_modules():  # here we disable weight_decay
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    if grad_clip > 0:
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0, 'clip_grad_max_norm':grad_clip, 'clip_grad_norm_type':2.0},
        ]
    else:
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

    return optim_groups
