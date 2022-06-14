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
from flowvision.layers.blocks.boxes import box_area


def box_cxcywh_to_xyxy(x):
    # TODO (ziqiu chi): unbind does not support global tensor
    # https://github.com/Oneflow-Inc/libai/pull/260#issuecomment-1153500398
    is_global = x.is_global
    if is_global:
        sbp, placement = x.sbp, x.placement
        x = x.to_local()
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)]
    box = flow.stack(b, dim=-1)
    if is_global:
        return box.to_global(sbp=sbp, placement=placement)
    return box

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return flow.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # NOTE: flow.max cannot min/max between diff dtype
    # NOTE: dim expand version leads the backward bug: Check failed: !broadcast_axis_vec.empty()
    # TODO(ziqiu chi): https://github.com/Oneflow-Inc/libai/issues/288#issuecomment-1144587028
    # lt = flow.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # rb = flow.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    if boxes1.shape[0] == 1 and boxes2.shape[0] == 1:
        lt = flow.max(boxes1[:, :2], boxes2[:, :2]).unsqueeze(1)  
        rb = flow.min(boxes1[:, 2:], boxes2[:, 2:]).unsqueeze(1)  
        
    else:
        lt = flow.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = flow.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]        
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union

    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    iou, union = box_iou(boxes1, boxes2)  # min/max op
    
    # TODO(ziqiu chi): https://github.com/Oneflow-Inc/libai/issues/288#issuecomment-1144587028
    # lt = flow.min(boxes1[:, None, :2], boxes2[:, :2])
    # rb = flow.max(boxes1[:, None, 2:], boxes2[:, 2:])
    if boxes1.shape[0] == 1 and boxes2.shape[0] == 1:
        lt = flow.min(boxes1[:, :2], boxes2[:, :2]).unsqueeze(1)  
        rb = flow.max(boxes1[:, 2:], boxes2[:, 2:]).unsqueeze(1)  
        
    else:
        lt = flow.min(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = flow.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]   
        
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return flow.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = flow.arange(0, h, dtype=flow.float)
    x = flow.arange(0, w, dtype=flow.float)
    y, x = flow.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return flow.stack([x_min, y_min, x_max, y_max], 1)
