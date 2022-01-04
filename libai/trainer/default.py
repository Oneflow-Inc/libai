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
import os
import oneflow as flow
from oneflow import nn
from typing import Callable
from libai.trainer.trainer import TrainerBase, EagerTrainer, GraphTrainer
from libai.utils import distributed as dist
from libai.utils.logger import setup_logger
from libai.utils.events import CommonMetricPrinter, JSONWriter
from libai.trainer import hooks
from libai.utils.checkpoint import Checkpointer
from libai.optim import build_optimizer
from libai.scheduler import build_lr_scheduler


def default_setup(cfg):
    """
    Perform some basic common setups at the beginning of a job, including:
    1. Set up the libai logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    Args:
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.output_dir
    if dist.is_main_process() and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rank = dist.get_rank()
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info(
        "Rank of current process: {}. World size: {}".format(
            rank, dist.get_world_size()
        )
    )

    flow.boxing.nccl.set_fusion_threshold_mbytes(cfg.nccl_fusion_threshold_mb)
    flow.boxing.nccl.set_fusion_max_ops_num(cfg.nccl_fusion_max_ops)
    flow.boxing.nccl.enable_use_compute_stream(True)


class DefaultTrainer(TrainerBase):
    """
    A trainer with default training logic. Compared to `TrainerBase`, it
    contains the following logic in addition:
    1. Create model, optimizer, scheduler, dataloader from the given config.
    2. Load a checkpoint or `cfg.MODEL.WEIGHTS`, if exists.
    3. Register a few common hooks.
    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`TrainerBase` are too much for research.
    The code of this class has been annotated about restrictive assumptions it made.
    When they do not work for you, you're encouraged to:
    1. Overwrite methods of this class, OR:
    2. Use :class:`TrainerBase`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.
    Also note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in libai.
    To obtain more stable behavior, write your own training logic with other public APIs.
    Attributes:
        scheduler:
        checkpointer:
        cfg (CfgNode):
    Examples:
    .. code-block:: python
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        self.cfg = cfg
        logger = logging.getLogger("libai")
        if not logger.isEnabledFor(
            logging.INFO
        ):  # setup_logger is not called for LibaiLM
            setup_logger()

        self.model = self.build_model(cfg)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.lr_scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = Checkpointer(
            # Assume you want to save checkpoints together with logs/statistics
            cfg,
            self.model,
            cfg.output_dir,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )

        if cfg.load is not None:
            self.resume_or_load()
            cfg.iteration = cfg.start_iter
        else:
            cfg.iteration = 0

        # Assume these objects must be constructed in this order.
        (
            self.train_data_iterator,
            self.valid_data_iterator,
            self.test_data_iterator,
        ) = self.build_train_valid_test_loader_loader(cfg)

        if cfg.mode == "graph":
            train_graph, eval_graph = self.build_graph(
                cfg, self.model, self.optimizer, self.lr_scheduler
            )
            # train_graph.debug(0)
            self._trainer = GraphTrainer(train_graph, self.train_data_iterator)
        elif cfg.mode == "eager":
            self._trainer = EagerTrainer(
                self.model, self.train_data_iterator, self.optimizer, self.lr_scheduler
            )
        else:
            raise NotImplementedError

        self.start_iter = cfg.iteration
        self.global_batch_size = cfg.global_batch_size
        self.max_iter = cfg.train_iters
        self._train_data = None

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        checkpoint = self.checkpointer.resume_or_load(self.cfg.load, resume=resume)

        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iter", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.save_interval),
        ]
        if dist.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(
                hooks.PeriodicWriter(self.build_writers(), self.cfg.log_interval)
            )
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.
        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        It is now implemented by:
        .. code-block:: python
            return [
                CommonMetricPrinter(self.global_batch_size, self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]
        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.global_batch_size, self.max_iter),
            JSONWriter(os.path.join(self.cfg.output_dir, "metrics.json")),
        ]

    def train(self):
        """
        Run training.
        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)

    def run_step(self, get_batch: Callable):
        self._trainer.iter = self.iter
        self._trainer.run_step(get_batch)

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            flow.nn.Module:
        It now calls :func:`libai.layers.build_model`.
        Overwrite it if you'd like a different model.
        """
        # TODO: import build_model from other utils
        # model = build_model(cfg)
        model = None
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_graph(cls, cfg, model, optimizer, lr_scheduler):
        # TODO: import build_graph from other utils
        return None
        # return build_graph(cfg, model, optimizer, lr_scheduler)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`libai.optim.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg.optim, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`libai.scheduler.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg.scheduler, optimizer)

    @classmethod
    def build_train_valid_test_loader_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`libai.data.build_reid_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        logger = logging.getLogger(__name__)
        logger.info("Prepare training set")
        # TODO: import build_train_valid_test_data_iterators from other utils
        return [None], [None], [None]
        # return build_train_valid_test_data_iterators(cfg)
