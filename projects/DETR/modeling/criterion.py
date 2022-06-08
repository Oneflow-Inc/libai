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


from numpy import place
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
from oneflow.env import get_world_size

from DETR.datasets.coco_dataloader import nested_tensor_from_tensor_list

from utils import box_ops
from utils.misc import accuracy, interpolate
from .segmentation import dice_loss, sigmoid_focal_loss


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

    def loss_labels(self, outputs, targets, indices, num_boxes, sbp, placement):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        target_classes = flow.full(src_logits.shape[:2], self.num_classes, dtype=flow.int64)
      
        target_classes_o = flow.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # .to(device=src_logits.device)
        
        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        idx = batch_idx, src_idx
        
        target_classes[idx] = target_classes_o
        target_classes = target_classes.to(device=src_logits.device)
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), 
            target_classes, 
            self.empty_weight.to(device=src_logits.device))
        losses = {'loss_ce': loss_ce.to(dtype=flow.float32).to_global(sbp=sbp, placement=placement)}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, sbp, placement):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = flow.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        l1_loss = nn.L1Loss(reduction="none")
        loss_bbox = l1_loss(src_boxes, target_boxes)

        losses = {}
        losses['loss_bbox'] = (loss_bbox.sum() / num_boxes).to(dtype=flow.float32).to_global(sbp=sbp, placement=placement)

        loss_giou = 1 - flow.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        
        losses['loss_giou'] = (loss_giou.sum() / num_boxes).to(dtype=flow.float32).to_global(sbp=sbp, placement=placement)
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, sbp, placement):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes).to_global(sbp=sbp, placement=placement),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes).to_global(sbp=sbp, placement=placement),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # NOTE: flow does not support flow.full_like
        # batch_idx = flow.cat([flow.full_like(src, i) for i, (src, _) in enumerate(indices)])
        batch_idx = flow.cat([flow.full(src.size(),i).to(dtype=src.dtype, device=src.device) for i, (src, _) in enumerate(indices)])
        src_idx = flow.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = flow.cat([flow.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = flow.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, sbp, placement):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, sbp, placement)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # outputs["pred_logits"].shape -> oneflow.Size([bsz, num_queries, 92])  
        # outputs["pred_boxes"].shape -> oneflow.Size([bsz, num_queries, 4])
        sbp, placement = outputs["pred_logits"].sbp, outputs["pred_logits"].placement
        # Switch outputs to local mode
        outputs = {k: v.to_local().to(device="cuda:0") for k, v in outputs.items() if k != "aux_outputs"}
        if "aux_outputs" in outputs.keys():
            for i in range(outputs["aux_outputs"]):
                for k, v in outputs["aux_outputs"][i].items():
                    outputs["aux_outputs"][i][k] = v.to_local().to(device="cuda:0")
        
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = flow.as_tensor([num_boxes], dtype=flow.float64)
        
        # Need get_world_size() in model parallel ?
        num_boxes = flow.clamp(num_boxes, min=1).item()
        # num_boxes = flow.clamp(num_boxes / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, sbp, placement))
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses