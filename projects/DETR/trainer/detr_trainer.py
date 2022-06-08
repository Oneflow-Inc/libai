'''
Author: hihippie chiziiqiu0923@gmail.com
Date: 2022-05-12 15:39:00
LastEditors: hihippie chiziiqiu0923@gmail.com
LastEditTime: 2022-05-26 13:49:50
FilePath: /libai/projects/DETR/trainer/detr_trainer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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

import time
from typing import Callable

from libai.engine.trainer import TrainerBase


class DetrEagerTrainer(TrainerBase):

    def __init__(self, model, data_loader, optimizer, grad_acc_steps=1):
        """
        Args:
            model: a flow.nn.Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a flow optimizer.
        """
        super().__init__()

        # We set the model to training mode in the trainer.
        # However it's valid to train a model that's in eval mode.
        # If you want your model (or a submodule of it) to behave
        # like evaluation during training, you can overwrite its train() method.

        model.train()
        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer
        self.grad_acc_steps = grad_acc_steps

    def run_step(self, get_batch: Callable):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        # If you want to do something with the data, you can wrap the dataloader.
        data = next(self._data_loader_iter)
        data = get_batch(data)
        data_time = time.perf_counter() - start
        loss_dict, _ = self.model(data)
        weight_dict = self.model.criterion.weight_dict
        losses = 0.
        for k in loss_dict.keys():
            if k in weight_dict:
                losses += loss_dict[k] * weight_dict[k]
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        losses.backward()
        self.write_metrics(loss_dict, data_time)
        
        if (self.iter + 1) % self.grad_acc_steps == 0:
            self.optimizer.clip_grad()
            self.optimizer.step()
            self.optimizer.zero_grad()