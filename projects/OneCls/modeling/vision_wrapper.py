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
import oneflow.nn as nn
from flowvision.layers import DropPath as vision_DropPath
from flowvision.models import ModelCreator

import libai.utils.distributed as dist
from libai.layers import DropPath, LayerNorm, Linear


class VisionModel(nn.Module):
    """
    Wrap the model from flowvision to be compatible with LiBai

    Args:
        model_name (str): model to be used for training.
        pretrained (bool): load the pretrained weight or not.
        num_classes (int): number of classes to be predicted.
    """

    def __init__(self, model_name, pretrained=False, num_classes=1000, loss_func=None, **kwargs):
        super().__init__()
        model = ModelCreator.create_model(
            model_name=model_name, pretrained=pretrained, num_classes=num_classes, **kwargs
        )
        self.model = self.libai_wrapper(model)
        # Loss func
        self.loss_func = nn.CrossEntropyLoss() if loss_func is None else loss_func

    def libai_wrapper(self, module):
        res = module
        if isinstance(module, vision_DropPath):
            res = DropPath(drop_prob=module.drop_prob)
        elif isinstance(module, nn.Linear):
            has_bias = True if module.bias is not None else False
            res = Linear(
                in_features=module.in_features, out_features=module.out_features, bias=has_bias
            )
            if has_bias:
                res.bias.data = module.bias.data.clone().to_global(
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), 
                    placement=res.bias.placement,
                )
            res.weight.data = module.weight.data.clone().to_global(
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), 
                placement=res.weight.placement,
            )
        elif isinstance(module, nn.LayerNorm):
            res = LayerNorm(
                normalized_shape=module.normalized_shape,
                eps=module.eps,
                elementwise_affine=module.elementwise_affine,
            )
            if module.elementwise_affine:
                res.weight.data = module.weight.data.clone().to_global(
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), 
                    placement=res.weight.placement,
                )
                res.bias.data = module.bias.data.clone().to_global(
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), 
                    placement=res.bias.placement,
                )
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = self.libai_wrapper(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

    def forward(self, images, labels=None):
        """

        Args:
            images (flow.Tensor): training samples.
            labels (flow.LongTensor, optional): training targets

        Returns:
            dict:
                A dict containing :code:`loss_value` or :code:`logits`
                depending on training or evaluation mode.
                :code:`{"losses": loss_value}` when training,
                :code:`{"prediction_scores": logits}` when evaluating.
        """
        x = self.model(images)

        if labels is not None and self.training:
            losses = self.loss_func(x, labels)
            return {"losses": losses}
        else:
            return {"prediction_scores": x}
