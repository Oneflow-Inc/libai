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

from libai.config import instantiate
from libai.utils.registry import Registry

SCHEDULER_REGISTRY = Registry("Scheduler")
SCHEDULER_REGISTRY.__doc__ = """
Registry for lr scheduler, i.e. WarmupCosineLR

The registered object will be called with `obj(cfg)` 
and expected to return a `flow.optim.lr_scheduler._LRScheduler` object.
"""

def build_lr_scheduler(cfg, optimizer):
    """ Build learning rate scheduler, defined by ``cfg.optim.lr_scheduler``.
    """
    if "_target_" in cfg.optim.lr_scheduler:
        scheduler = instantiate(cfg.optim.lr_scheduler)
    else:
        scheduler_name = cfg.optim.lr_scheduler.name
        scheduler = SCHEDULER_REGISTRY.get(scheduler_name)(optimizer, **cfg.optim.lr_scheduler.scheduler_cfg)
    return scheduler
    