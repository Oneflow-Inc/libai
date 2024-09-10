# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
# Copyright 2023-present the HuggingFace Inc. team.
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

import re
from typing import List

import oneflow as flow

COMMON_LAYERS_PATTERN = ["layers", "h", "block", "blocks", "layer"]


def check_target_module_exists(config, key: str) -> bool | re.Match[str] | None:
    """A helper method to check if the passed module's key name matches
       any of the target modules in the adapter_config.

    Args:
        config (`LoraConfig` | `LycorisConfig`): A config to match
        target modules from
        key (`str`): A key to search any matches in config

    Returns:
        `bool` | `re.Match[str]` | `None`: True of match object if key matches any
        target modules from config, False or None if no match found
    """
    if isinstance(config.target_modules, str):
        target_module_found = re.fullmatch(config.target_modules, key)
    else:
        target_module_found = key in config.target_modules or any(
            key.endswith(f".{target_key}") for target_key in config.target_modules
        )
        is_using_layer_indexes = getattr(config, "layers_to_transform", None) is not None
        layer_indexing_pattern = getattr(config, "layers_pattern", None)

        if is_using_layer_indexes and target_module_found:
            layers_pattern = (
                COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
            )
            layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

            for pattern in layers_pattern:
                layer_index = re.match(r".*.{" + pattern + r"}\.(\d+)\.*", key)
                if layer_index is not None:
                    layer_index = int(layer_index.group(1))
                    if isinstance(config.layers_to_transform, int):
                        target_module_found = layer_index == config.layers_to_transform
                    else:
                        target_module_found = layer_index in config.layers_to_transform

                    break
                else:
                    target_module_found = False
    return target_module_found


def _get_submodules(model, key):

    parent = _get_submodule(model, ".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = _get_submodule(model, key)
    return parent, target, target_name


def _get_submodule(model, target):
    if target == "":
        return model

    atoms: List[str] = target.split(".")
    mod: flow.nn.Module = model

    for item in atoms:

        if not hasattr(mod, item):
            raise AttributeError(mod._get_name() + " has no " "attribute `" + item + "`")

        mod = getattr(mod, item)

        if not isinstance(mod, flow.nn.Module):
            raise AttributeError("`" + item + "` is not " "an nn.Module")

    return mod
