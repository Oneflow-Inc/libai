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

from typing import Callable, Optional

import oneflow as flow
from libai.config.configs.common.data.coco import NestedTensor

from libai.data import Instance


from libai.engine.default import DefaultTrainer
from trainer.detr_trainer import DetrEagerTrainer


class DetrDefaultTrainer(DefaultTrainer):

    def __init__(self, cfg):

        super().__init__(cfg)

        self.model.max_iter = cfg.train.train_iter

        self._trainer = DetrEagerTrainer(
            self.model, self.train_loader, self.optimizer, cfg.train.num_accumulation_steps
        )

    @classmethod
    def get_batch(cls, data: Instance, mixup_func: Optional[Callable] = None):
        """
        Convert batched local tensor to distributed tensor for model step running.

        If you want to do something with batched data before model, (e.g. mixup),
        you can rewrite this function.
        """
        if isinstance(data, flow.utils.data._utils.worker.ExceptionWrapper):
            data.reraise()

        if mixup_func is not None:
            images, labels = mixup_func(
                data.get("images").tensor.cuda(),
                data.get("labels").tensor.cuda(),
            )
            data.get("images").tensor = images
            data.get("labels").tensor = labels

        ret_dict = {}
        for key, value in data.get_fields().items():
            # NOTE: deal with nestedtensor
            import pdb
            pdb.set_trace()
            tensors, mask = value.tensors, value.mask
            tensors.to_global()
            mask.to_global()
            ret_dict[key] = NestedTensor(tensors, mask)
        return ret_dict