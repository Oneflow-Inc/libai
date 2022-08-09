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


from libai.config import LazyCall

from ...modeling.backbone import Backbone, Joiner
from ...modeling.matcher import HungarianMatcher
from ...modeling.transformer import Transformer
from ...modeling.detr import DETR
from ...modeling.criterion import SetCriterion
from ...modeling.position_encoding import PositionEmbeddingLearned, PositionEmbeddingSine


def build_criterion(args):
    num_classes = 20 if args.dataset_file != "coco" else 91
    weight_dict = {"loss_ce": 1, "loss_bbox": args.bbox_loss_coef, "loss_giou": args.giou_loss_coef}

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes"]

    return SetCriterion(
        num_classes=num_classes,
        matcher=HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
        ),
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
    )


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ("v2", "sine"):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding


def build_backbone(args):
    position_embedding = build_position_encoding(args=args)
    # TODO (ziqiu chi): return_interm_layers works for segmentation task
    return_interm_layers = args.masks
    backbone = Backbone(
        name=args.backbone,
        train_backbone=args.train_backbone,
        return_interm_layers=return_interm_layers,
        dilation=args.dilation,
    )
    model = Joiner(backbone=backbone, position_embedding=position_embedding)
    model.num_channels = backbone.num_channels
    return model


def build(args):
    """
    Build the DETR model for detection
    """

    num_classes = 20 if args.dataset_file != "coco" else 91

    backbone = LazyCall(build_backbone)(args=args)
    transformer = LazyCall(build_transformer)(args=args)
    criterion = LazyCall(build_criterion)(args=args)

    model = LazyCall(DETR)(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        criterion=criterion,
    )

    return model
