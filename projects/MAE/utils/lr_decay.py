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


# --------------------------------------------------------
# References:
# mae: https://github.com/facebookresearch/mae/blob/main/util/lr_decay.py
# --------------------------------------------------------


def get_layer_wise_lrd_overrides(model, learning_rate, layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Modified from BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    overrides = {}
    num_layers = len(model.blocks) + 1
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    
    for name, param in model.named_parameters():
        layer_idx = get_layer_idx_for_vit(name, num_layers)
        overrides[name] = {
            "lr": learning_rate * layer_scales[layer_idx]
        }
    
    return overrides


def get_layer_idx_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers