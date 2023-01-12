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

import logging

import oneflow as flow
from oneflow import nn

from libai.utils import distributed as dist

logger = logging.getLogger(__name__)


class CTRGraph(nn.Graph):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module = None,
        optimizer: flow.optim.Optimizer = None,
        lr_scheduler: flow.optim.lr_scheduler = None,
        grad_scaler = None,
        fp16=False,
        is_train=True,
    ):
        super().__init__()

        self.model = model
        self.is_train = is_train

        if is_train:
            self.loss = loss
            self.add_optimizer(optimizer, lr_sch=lr_scheduler)
            if fp16:
                self.config.enable_amp(True)
                self.set_grad_scaler(grad_scaler)

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_cast_scale(True)

    def build(self, batch_dict):
        if self.is_train:
            logger.info(
                "Start compling the train graph which may take some time. "
                "Please wait for a moment ..."
            )
            logits = self.model(batch_dict)
            loss = self.loss(logits, batch_dict['label'].to("cuda"))
            reduce_loss = flow.mean(loss)
            reduce_loss.backward()
            loss_dict = {"bce_loss": loss}
            #loss_dict = {
            #    "bce_loss": loss.to_global(
            #        placement=flow.placement(
            #            "cpu", ranks=[0] if loss.placement.ranks.ndim == 1 else [[0]]
            #        ),
            #        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            #    )
            #}
            return loss_dict
        else:
            logger.info(
                "Start compling the eval graph which may take some time. "
                "Please wait for a moment ..."
            )
            predicts = self.models(batch_dict)
            return predicts.sigmoid()

