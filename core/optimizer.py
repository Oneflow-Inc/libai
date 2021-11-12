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

import oneflow as flow
from core.modules import ParallelEmbedding, ColumnParallelLinear, RowParallelLinear, LayerNorm, Embedding, ParallelEmbedding, PositionalEmbedding


def build_grad_scaler(args):
    if args.loss_scale is not None:
        grad_scaler = flow.amp.StaticGradScaler(args.loss_scale)
    elif args.initial_loss_scale is not None:
        grad_scaler = flow.amp.GradScaler(
            init_scale=args.initial_loss_scale,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=args.loss_scale_window,
        )
    else:
        grad_scaler = None

    return grad_scaler


def build_lr_scheduler(args, optimizer):
    assert args.lr_decay_style in ("none", "cosine")

    if args.lr_decay_style == "none":
        return None

    if args.lr_decay_iters is None:
        return None

    lr_decay_alpha = args.min_lr / args.lr
    lr_scheduler = flow.optim.lr_scheduler.CosineDecayLR(
        optimizer, decay_steps=args.lr_decay_iters, alpha=lr_decay_alpha,
    )

    if args.lr_warmup_iters is not None and args.lr_warmup_iters > 0:
        lr_scheduler = flow.optim.lr_scheduler.WarmUpLR(
            lr_scheduler,
            warmup_factor=0,
            warmup_iters=args.lr_warmup_iters,
            warmup_method="linear",
        )

    return lr_scheduler


def _get_params_for_weight_decay_optimization(model):
    """For layernorm module and bias, don't apply weight decay.
    For other module, apply weight decay during training.
    """
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}

    for module in model.modules():
        if isinstance(module, LayerNorm):
            no_weight_decay_params["params"].extend(
                [p for p in list(module.parameters(recurse=False)) if p is not None]
            )
        else:
            weight_decay_params["params"].extend(
                [p for n, p in list(module.named_parameters(recurse=False)) if p is not None and n != "bias"]
            )
            no_weight_decay_params["params"].extend(
                [p for n, p in list(module.named_parameters(recurse=False)) if p is not None and n == "bias"]
            )

    return weight_decay_params, no_weight_decay_params


def _set_clip_grad_for_param_groups(param_groups, clip_grad):
    if int(clip_grad) == 1:
        for group in param_groups:
            group["clip_grad_max_norm"] = 1.0
            group["clip_grad_norm_type"] = 2.0


def build_optimizer(args, model):
    param_groups = _get_params_for_weight_decay_optimization(model)
    _set_clip_grad_for_param_groups(param_groups, args.clip_grad)

    if args.optimizer == "adamw":
        optimizer = flow.optim.AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif args.optimizer == "sgd":
        optimizer = flow.optim.SGD(param_groups, lr=args.lr)
    else:
        raise NotImplementedError("not supported yet")

    return optimizer
