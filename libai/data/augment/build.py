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

import flowvision as vision

from libai.utils.registry import Registry

AUGMENT_REGISTRY = Registry("Optimizer")
AUGMENT_REGISTRY.__doc__ = """
Registry for augmentation for vision task, i.e. Auto-Augment

The registered object will be called with `obj(cfg)` 
and expected to return a `flowvision.transforms` object.
"""

AUGMENT_REGISTRY.register(vision.data.Mixup)
AUGMENT_REGISTRY.register(vision.data.AutoAugment)
AUGMENT_REGISTRY.register(vision.data.RandAugment)
AUGMENT_REGISTRY.register(vision.data.AugMixAugment)
AUGMENT_REGISTRY.register(vision.data.create_transform)
AUGMENT_REGISTRY.register(vision.data.transforms_imagenet_eval)
AUGMENT_REGISTRY.register(vision.data.transforms_imagenet_train)
