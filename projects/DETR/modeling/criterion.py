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

from modeling.cross_entropy import ParallelCrossEntropyLoss
from utils import box_ops


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = flow.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.l1_loss = nn.L1Loss(reduction="none")
        self.cross_entropy = ParallelCrossEntropyLoss()

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        target_classes_o = flow.cat([t[J] for t, (_, J) in zip(targets["labels"], indices)])
        target_classes_o = target_classes_o.to_local()
        target_classes = flow.full(src_logits.shape[:2], self.num_classes, dtype=flow.int64)
        idx = self._get_src_permutation_idx(indices)
        target_classes[idx] = target_classes_o
        target_classes = target_classes.to_global(sbp=src_logits.sbp, placement=src_logits.placement)
        loss_ce = self.cross_entropy(
            src_logits, 
            target_classes,
            self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = flow.cat([t[i] for t, (_, i) in zip(targets["boxes"], indices)], dim=0)
        loss_bbox = self.l1_loss(src_boxes, target_boxes).sum()
        losses = {}
        losses['loss_bbox'] = (loss_bbox / num_boxes)
        loss_giou = 1 -  box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)).diag()
        losses['loss_giou'] = (loss_giou.sum() / num_boxes)
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # NOTE: flow does not support flow.full_like
        batch_idx = flow.cat([flow.full(src.size(),i).to(dtype=src.dtype, device=src.device) for i, (src, _) in enumerate(indices)])
        src_idx = flow.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = flow.cat([flow.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = flow.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
                      For coco dataset:
                      outputs["pred_logits"].shape: oneflow.Size([bsz, num_queries, 92])  
                      outputs["pred_boxes"].shape: oneflow.Size([bsz, num_queries, 4])
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_boxes = sum(len(t["labels"][t["target_mask"]]) for t in targets)
        num_boxes = targets["target_orig_size"].sum()
        num_boxes = flow.as_tensor([num_boxes], dtype=flow.float64)
        num_boxes = flow.clamp(num_boxes, min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute.
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


@flow.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        # BUG: 100 - [flow.zeros_like(target)] -> []
        # 100 - [flow.zeros([], sbp=target.sbp, placement=target.placement)] -> 100
        # return [flow.zeros_like(target)]
        return [flow.zeros([], sbp=target.sbp, placement=target.placement)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res