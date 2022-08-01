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
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer
        self.grad_acc_steps = grad_acc_steps
        self.mem = []
        
    def run_step(self, get_batch: Callable,  input_placement_device: str = "cuda"):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data = get_batch(data)
        data_time = time.perf_counter() - start
        loss_dict, _ = self.model(data)
        weight_dict = self.model.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        losses.backward()
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        self.write_metrics(loss_dict_scaled, data_time)
        if (self.iter + 1) % self.grad_acc_steps == 0:
            self.optimizer.clip_grad()
            self.optimizer.step()
            self.optimizer.zero_grad()                                