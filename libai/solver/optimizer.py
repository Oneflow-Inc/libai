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

"""
Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/solver/build.py
"""

import copy
import itertools
import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

import oneflow as flow
import oneflow.nn as nn

from libai.layers import LayerNorm

_GradientClipperInput = Union[flow.Tensor, Iterable[flow.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]

class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"

def _create_gradient_clipper(clip_type: str, clip_value: float, norm_type: float) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """

    def clip_grad_norm(p: _GradientClipperInput):
        nn.utils.clip_grad_norm_(p, clip_value, norm_type)
    
    def clip_grad_value(p: _GradientClipperInput):
        nn.utils.clip_grad_value_(p, clip_value)
    
    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(clip_type)]


def _generate_optimizer_class_with_gradient_clipping(
    optimizer: Type[flow.optim.Optimizer],
    *,
    per_param_clipper: Optional[_GradientClipper] = None,
    global_clipper: Optional[_GradientClipper] = None,
) -> type[flow.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """
    assert (
        per_param_clipper is None or global_clipper is None
    ), "Not allowed to use both per-parameter clipping and global clipping"

    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
    
        else:
            # global clipper for future use with detr
            # (https://github.com/facebookresearch/detr/pull/287)
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            global_clipper(all_params)
        super(type(self), self).step(closure)
    
    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(
    enable_clip: bool, clip_type: str, clip_value: float, norm_type: float, optimizer: Type[flow.optim.Optimizer]
) -> Type[flow.optim.Optimizer]:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer type to become a new dynamically created class OptimizerWithGradientClip
    that inherits the given optimizer and overrides the `step` method to
    include gradient clipping.

    Args:
        enable_clip: bool. Enable gradient clipping or not.
        clip_type: float. Type of gradient clippint, choose from ["value" and "norm"].
        clip_value: float. Maximum absolute value used for clipping gradients.
        norm_type: float. Floating point number p for L-p norm to be used with the "norm" gradient clippint type;
        optimizer: type. A subclass of torch.optim.Optimizer

    Return:
        type: either the input `optimizer` (if gradient clipping is disabled), or
            a subclass of it with gradient clipping included in the `step` method.
    """
    if not enable_clip:
        return optimizer
    if isinstance(optimizer, flow.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, flow.optim.Optimizer), optimizer
        optimizer_type = optimizer

    grad_clipper = _create_gradient_clipper(clip_type, clip_value, norm_type)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        optimizer_type, per_param_clipper=grad_clipper
    )
    if isinstance(optimizer, flow.optim.Optimizer):
        optimizer.__class__ = OptimizerWithGradientClip  # a bit hacky, not recommended
        return optimizer
    else:
        return OptimizerWithGradientClip


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
    Get default param list for optimizer, with suport for a few types of overrides. If no overrides needed, this is equivalent to `model.parameters()`.
    
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
        flow.optim.AdamW(get_default_optimizer_params(model, weight_decay_norm=0, weight_decay_bias=0),
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