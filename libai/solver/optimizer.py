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

_GradientClipperInput = Union[flow.Tensor, Iterable[flow.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]

class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"

def _create_gradient_clipper(cfg) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = copy.deepcopy(cfg)

    # TODO: add optim.clip_gradient cfg into libai's default config file
    def clip_grad_norm(p: _GradientClipperInput):
        nn.utils.clip_grad_norm_(p, cfg.optim.clip_gradient.clip_value, cfg.optim.clip_gradient.norm_type)
    
    def clip_grad_value(p: _GradientClipperInput):
        nn.utils.clip_grad_value_(p, cfg.optim.clip_gradient.clip_value)
    
    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.optim.clip_gradient.clip_type)]


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
    cfg, optimizer: Type[flow.optim.Optimizer]
) -> Type[flow.optim.Optimizer]:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer type to become a new dynamically created class OptimizerWithGradientClip
    that inherits the given optimizer and overrides the `step` method to
    include gradient clipping.

    Args:
        cfg: configuration options
        optimizer: type. A subclass of torch.optim.Optimizer

    Return:
        type: either the input `optimizer` (if gradient clipping is disabled), or
            a subclass of it with gradient clipping included in the `step` method.
    """
    if not cfg.optim.clip_gradients.enable:
        return optimizer
    if isinstance(optimizer, flow.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, flow.optim.Optimizer), optimizer
        optimizer_type = optimizer

    grad_clipper = _create_gradient_clipper(cfg.SOLVER.CLIP_GRADIENTS)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        optimizer_type, per_param_clipper=grad_clipper
    )
    if isinstance(optimizer, flow.optim.Optimizer):
        optimizer.__class__ = OptimizerWithGradientClip  # a bit hacky, not recommended
        return optimizer
    else:
        return OptimizerWithGradientClip