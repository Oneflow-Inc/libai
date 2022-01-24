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
import time
import weakref
from typing import Callable, List, Mapping

import numpy as np
import oneflow as flow

from libai.utils import distributed as dist
from libai.utils.events import EventStorage, get_event_storage


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()
    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).
        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.
           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.
    """

    trainer: "TrainerBase" = None
    """
    A weak reference to the trainer object. Set by the trainer when the hook is registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """

    def after_train(self):
        """
        Called after the last iteration.
        """

    def before_step(self):
        """
        Called before each iteration.
        """

    def after_step(self):
        """
        Called after each iteration.
        """


class TrainerBase:
    """
    Base class for iterative trainer with hooks.
    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.
    Attributes:
        iter(int): the current iteration.
        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.
        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self):
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage

    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.
        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        self.storage.iter = self.iter + 1
        self.storage.samples = (self.iter + 1) * self.cfg.train.global_batch_size

        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, flow.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {k: dist.tton(v, local_only=False) for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # TODO: Gather metrics among all workers for logging
        # all_metrics_dict = dist.gather(metrics_dict)
        all_metrics_dict = metrics_dict

        # dist_util = dist.get_dist_util()
        # if (dist_util.is_pipeline_model_parallel() and dist.is_last_process()) or (
        #     not dist_util.is_pipeline_model_parallel() and dist.is_main_process()
        # ):
        if dist.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            # data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            data_time = all_metrics_dict.pop("data_time")
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            # metrics_dict = {
            #     k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            # }
            metrics_dict = all_metrics_dict
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={storage.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)


class EagerTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
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

    def __init__(self, model, data_loader_iter, optimizer):
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
        self._data_loader_iter = iter(data_loader_iter)
        self.optimizer = optimizer

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

        # If you want to do something with the losses, you can wrap the model.

        losses = self.model(*data)
        loss_dict = {"total_loss": losses}

        # If you need to accumulate gradients or do something similar, you can
        # wrap the optimizer with your custom `zero_grad()` method.

        self.optimizer.zero_grad()
        losses.backward()

        self.write_metrics(loss_dict, data_time)

        # If you need gradient clipping/scaling or other processing, you can
        # wrap the optimizer with your custom `step()` method. But it is
        # suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4

        self.optimizer.step()


class GraphTrainer(TrainerBase):
    def __init__(self, graph, data_loader_iter):
        super().__init__()

        graph.model.train()
        self._data_loader_iter = iter(data_loader_iter)
        self.graph = graph

    def run_step(self, get_batch: Callable):
        """
        Implement the standard training logic described above.
        """
        assert self.graph.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        # If you want to do something with the data, you can wrap the dataloader.
        data = next(self._data_loader_iter)
        data = get_batch(data)
        data_time = time.perf_counter() - start

        # If you want to do something with the losses, you can wrap the model.

        losses = self.graph(*data)
        loss_dict = {"total_loss": losses}

        self.write_metrics(loss_dict, data_time)
