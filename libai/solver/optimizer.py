"""
Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/solver/build.py
"""

import copy
import itertools
import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
from yacs.config import CfgNode

import oneflow as flow
import oneflow.nn as nn

_GradientClipperInput = Union[flow.Tensor, Iterable[flow.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]

class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"

def _create_gradient_clipper(cfg: CfgNode) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = copy.deepcopy(cfg)

    def clip_grad_norm(p: _GradientClipperInput):
        nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)
    
    def clip_grad_value(p: _GradientClipperInput):
        nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)
    
    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]


def _generate_optimizer_class_with_gradient_clippint(
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
