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

import oneflow as flow

from libai.data.structures import DistTensorData, Instance
from libai.engine.trainer import TrainerBase


class DetrEagerTrainer(TrainerBase):
    """
    A simple eager trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.
    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

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

        # NOTE: it is better to impl this in libai/data/datasets/coco.py, 
        # data = list(data)
        # images, labels = data[0], data[1][0]
        # images.tensors = DistTensorData(images.tensors, placement_idx=-1)
        # images.mask = DistTensorData(images.mask, placement_idx=-1)
        # for k,v in labels.items():
        #     labels[k] = DistTensorData(flow.tensor(labels[k]).long(), placement_idx=-1)
        # data = Instance(
        #     images = images,
        #     labels = labels
        # )
        data = get_batch(data, getattr(self.data_loader, "mixup_func", None))
        data_time = time.perf_counter() - start
        loss_dict = self.model(data["images"])
        
        # TODO: continue here
        losses = sum(loss_dict.values()) / self.grad_acc_steps

        losses.backward()

        self.write_metrics(loss_dict, data_time)

        if (self.iter + 1) % self.grad_acc_steps == 0:
            self.optimizer.clip_grad()
            self.optimizer.step()
            self.optimizer.zero_grad()