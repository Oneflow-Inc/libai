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
# DETR Model
# References:
# https://github.com/facebookresearch/detr/blob/main/models/backbone.py
# --------------------------------------------------------


from typing import Dict, List

import oneflow as flow
import oneflow.nn as nn 
import oneflow.nn.functional as F
import flowvision
from flowvision.models.layer_getter import IntermediateLayerGetter

from libai.config.configs.common.data.coco import NestedTensor
import libai.utils.distributed as dist

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from flowvision.misc.ops with added eps before rqsrt,
    without which any other models than flowvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", flow.ones(n), persistent=True)
        self.register_buffer("bias", flow.zeros(n), persistent=True)
        self.register_buffer("running_mean", flow.zeros(n), persistent=True)
        self.register_buffer("running_var", flow.ones(n), persistent=True)
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        placement = x.placement
        
        w = self.weight.reshape(1, -1, 1, 1).to_global(sbp=sbp, placement=placement)
        b = self.bias.reshape(1, -1, 1, 1).to_global(sbp=sbp, placement=placement)
        rv = self.running_var.reshape(1, -1, 1, 1).to_global(sbp=sbp, placement=placement)
        rm = self.running_mean.reshape(1, -1, 1, 1).to_global(sbp=sbp, placement=placement)
        
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
        
        # self.substitute = {}

    def forward(self, tensor_list: NestedTensor):

        xs = self.body(tensor_list.tensors.tensor)
            
        out: Dict[str, NestedTensor] = {}
        
        
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None

            mask = F.interpolate(m.tensor[None].float(), size=x.shape[-2:]).to(flow.bool)[0]

            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):

        backbone = getattr(flowvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=dist.is_main_process(), norm_layer=FrozenBatchNorm2d)
        
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(
        name=args.backbone, train_backbone=train_backbone, 
        return_interm_layers=return_interm_layers, dilation=args.dilation)
    model = Joiner(backbone=backbone, position_embedding=position_embedding)
    model.num_channels = backbone.num_channels
    return model
