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

import copy
import inspect
from collections import defaultdict
from typing import Any, Dict, List

import oneflow as flow

from libai.config import instantiate
from libai.layers import LayerNorm
from libai.utils.registry import Registry

OPTIMIZER_REGISTRY = Registry("Optimizer")
OPTIMIZER_REGISTRY.__doc__ = """
Registry for optimizer, i.e. SGD, AdamW

The registered object will be called with `obj(cfg)`
and expected to return a `flow.optim.Optimizer` object.
"""


def register_optimizer():
    flow_optimizers = []
    for module_name in dir(flow.optim):
        if module_name.startswith("__"):
            continue
        _optim = getattr(flow.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, flow.optim.Optimizer):
            OPTIMIZER_REGISTRY.register(_optim)
            flow_optimizers.append(module_name)
    return flow_optimizers


FLOW_OPTIMIZERS = register_optimizer()


def build_optimizer(cfg, model):
    if "_target_" in cfg:
        cfg.parameters.model = model
        optim = instantiate(cfg)
    else:
        optim_name = cfg.optim_name
        optim = OPTIMIZER_REGISTRY.get(optim_name)(
            get_default_optimizer_params(model, **cfg.param_cfg), **cfg.optim_cfg
        )
    return optim


def get_default_optimizer_params(
    model,
    base_lr=None,
    weight_decay=None,
    weight_decay_norm=None,
    weight_decay_bias=None,
    clip_grad_max_norm=None,
    clip_grad_norm_type=None,
    overrides=None,
):
    """
    Get default param list for optimizer, with suport for a few types of overrides.
    If no overrides needed, this is equivalent to `model.parameters()`.

    Arguments:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        weight_decay_bias: override weight decay for bias parameters
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.
    For common transformer models, ``weight_decay_norm,weight_decay_bias`` is usually set to 0.
    Example:
    ::
        flow.optim.AdamW(
            get_default_optimizer_params(model, weight_decay_norm=0, weight_decay_bias=0),
                       lr=0.01, weight_decay=1e-4)
    """
    if overrides is None:
        overrides = {}
    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay
    if clip_grad_max_norm is not None and clip_grad_norm_type is not None:
        defaults["clip_grad_max_norm"] = clip_grad_max_norm
        defaults["clip_grad_norm_type"] = clip_grad_norm_type
    bias_overrides = {}
    if weight_decay_bias is not None:
        bias_overrides["weight_decay"] = weight_decay_bias
    if len(bias_overrides):
        if "bias" in overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        overrides["bias"] = bias_overrides

    norm_module_types = (
        LayerNorm,
        flow.nn.BatchNorm1d,
        flow.nn.BatchNorm2d,
        flow.nn.BatchNorm3d,
        flow.nn.GroupNorm,
        flow.nn.InstanceNorm1d,
        flow.nn.InstanceNorm2d,
        flow.nn.InstanceNorm3d,
        flow.nn.FusedBatchNorm1d,
        flow.nn.FusedBatchNorm2d,
        flow.nn.FusedBatchNorm3d,
    )
    params = []
    memo = set()
    for module in model.modules():
        for model_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm
            hyperparams.update(overrides.get(model_param_name, {}))
            params.append({"params": [value], **hyperparams})
    return reduce_param_groups(params)


def _expand_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform parameter groups into per-parameter structure.
    Later items in `params` can overwrite parameters set in previous items.
    """
    ret = defaultdict(dict)
    for item in params:
        assert "params" in item
        cur_params = {x: y for x, y in item.items() if x != "params"}
        for param in item["params"]:
            ret[param].update({"params": [param], **cur_params})
    return list(ret.values())


def reduce_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reorganize the parameter groups and merge duplicated groups.
    The number of parameter groups needs to be as small as possible in order
    to efficiently use the PyTorch multi-tensor optimizer. Therefore instead
    of using a parameter_group per single parameter, we reorganize the
    parameter groups and merge duplicated groups. This approach speeds
    up multi-tensor optimizer significantly.
    """
    params = _expand_param_groups(params)
    groups = defaultdict(list)  # re-group all parameter groups by their hyperparams
    for item in params:
        cur_params = tuple((x, y) for x, y in item.items() if x != "params")
        groups[cur_params].extend(item["params"])
    ret = []
    for param_keys, param_values in groups.items():
        cur = {kv[0]: kv[1] for kv in param_keys}
        cur["params"] = param_values
        ret.append(cur)
    return ret
