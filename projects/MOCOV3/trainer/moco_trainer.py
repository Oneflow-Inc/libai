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

from libai.engine.trainer import EagerTrainer


class MoCoEagerTrainer(EagerTrainer):
    def run_step(self, get_batch: Callable):

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        # If you want to do something with the data, you can wrap the dataloader.
        data = next(self._data_loader_iter)
        data = get_batch(data, getattr(self.data_loader, "mixup_func", None))
        data_time = time.perf_counter() - start

        # update the moco_momentum per step
        loss_dict, m_dict = self.model(**data, cu_iter=self.iter, m=self.model.m)
        self.model.m = m_dict["m"]
        losses = sum(loss_dict.values()) / self.grad_acc_steps
        losses.backward()
        self.write_metrics(loss_dict, data_time)

        if (self.iter + 1) % self.grad_acc_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
