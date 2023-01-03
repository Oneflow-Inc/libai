# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates.
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
import math
import os
import time
from collections import OrderedDict
from typing import Callable, Optional

import oneflow as flow
from omegaconf import OmegaConf
from termcolor import colored

from libai.config import LazyConfig, instantiate, try_get_key
from libai.data import Instance
from libai.engine import hooks
from libai.engine.trainer import EagerTrainer, GraphTrainer, TrainerBase
from libai.evaluation import inference_on_dataset, print_csv_format
from libai.models import build_ctr_graph, build_model
from libai.optim import build_optimizer
from libai.scheduler import build_lr_scheduler
from libai.utils import distributed as dist
from libai.utils.checkpoint import Checkpointer
from libai.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from libai.utils.logger import setup_logger

# --------------------------------------------------------
# References:
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py
# --------------------------------------------------------


def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.formatters import Terminal256Formatter
    from pygments.lexers import Python3Lexer, YamlLexer

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


def ctr_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the libai logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Setup the distributed environment
    6. Backup the config to the output directory

    Args:
        args (argparse.NameSpace): the command line arguments to be logged
    """

    output_dir = try_get_key(cfg, "train.output_dir")
    if dist.is_main_process() and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cfg.train.resume = args.resume
    
    cfg.train.global_batch_size = cfg.dataloader.train.batch_size

    rank = dist.get_rank()
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, dist.get_world_size()))
    logger.info("Command line arguments: " + str(args))

    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(open(args.config_file, "r").read(), args.config_file),
            )
        )

    #dist.setup_dist_util(cfg.train.dist)

    if dist.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))

    flow.boxing.nccl.set_fusion_threshold_mbytes(
        try_get_key(cfg, "train.nccl_fusion_threshold_mb", default=16)
    )
    flow.boxing.nccl.set_fusion_max_ops_num(
        try_get_key(cfg, "train.nccl_fusion_max_ops", default=24)
    )


class CTRTrainer(TrainerBase):
    """
    A trainer with default training logic. Compared to `TrainerBase`, it
    also contains the following logic:

    1. Create model, optimizer, scheduler, dataloader from the given config.
    2. Load a checkpoint or `cfg.MODEL.WEIGHTS`, if exists.
    3. Register a few common hooks defined by the config.

    With standard features, it is created to simplify the **standard model training workflow** and
    reduce code boilerplate for users who only need the standard training workflow.

    It means this class makes **many assumptions** about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`TrainerBase` are too much for research.

    The code of this class has been annotated about restrictive assumptions it made.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`TrainerBase`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to ``tools/train_net.py``.

    Also note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in libai.
    To obtain more stable behavior, write your own training logic with other public APIs.


    Examples:

    .. code-block:: python

        trainer = CTRTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (Checkpointer):
        cfg (omegaconf.dictconfig.DictConfig):
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (omegaconf.dictconfig.DictConfig):
        """
        super().__init__()
        self.cfg = cfg
        logger = logging.getLogger("libai")

        # setup_logger is not called for LiBai
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()

        self.start_iter = 0
        if cfg.train.resume:
            save_file = os.path.join(cfg.train.output_dir, "last_checkpoint")
            try:
                with open(save_file, "r") as f:
                    last_saved = f.read().strip()
                assert (
                    last_saved != "model_final"
                ), "model training has finished, check your model in train.output_dir"
                self.start_iter = int(last_saved.split("_")[-1]) + 1
            except IOError:
                # If file doesn't exist, maybe because it has just been deleted.
                # We just set start_iter to 0.
                self.start_iter = 0
        #cfg.dataloader.consumed_samples = self.start_iter * cfg.train.batch_size
        self.train_loader, self.val_loader, self.test_loader = self.build_data_loader(cfg)

        if cfg.train.rdma_enabled:
            # set rdma
            flow.env.init_rdma()

        # Assume these objects must be constructed in this order.
        dist.synchronize()
        start_time = time.time()
        logger.info("> Start building model...")
        self.model = self.build_model(cfg)

        dist.synchronize()
        logger.info(
            ">>> done with building model. "
            "Building time: {:.3f} seconds".format(time.time() - start_time)
        )

        self.optimizer = self.build_optimizer(cfg, self.model)
        self.lr_scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        self.loss = flow.nn.BCEWithLogitsLoss(reduction="none").to("cuda")

        if cfg.graph.enabled:
            self.graph_train = self.build_graph(
                cfg, self.model, self.loss, self.optimizer, self.lr_scheduler, is_train=True
            )
            self.graph_eval = self.build_graph(cfg, self.model, is_train=False)
            self._trainer = GraphTrainer(self.graph_train, self.train_loader)
        else:
            self._trainer = EagerTrainer(self.model, self.train_loader, self.optimizer)

        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        if cfg.graph.enabled:
            self.checkpointer = Checkpointer(
                # Assume you want to save checkpoints together with logs/statistics
                self.model,
                cfg.train.output_dir,
                # In static graph mode, optimizer and scheduler state_dict will
                # be saved with graph.state_dict().
                graph=self.graph_train,
                # We print lr by `LRScheduler` hook, so we need to save/load eager lr_scheduler,
                # otherwise, lr will be reset to initial state when resuming training.
                lr_scheduler=self.lr_scheduler,
            )
        else:
            self.checkpointer = Checkpointer(
                # Assume you want to save checkpoints together with logs/statistics
                self.model,
                cfg.train.output_dir,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
            )

        # Loading checkpoint before dataloader construction, because
        # dataloader needs to know the consumed iterations from
        # the last breakpoint.
        #todo self.resume_or_load(cfg.train.resume)
        cfg.train.start_iter = self.start_iter
        self.max_iter = cfg.train.train_iter

        #todo self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.train.output_dir` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.train.load_weight`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file ``cfg.train.load_weight`` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        weight_path = self.cfg.train.load_weight
        assert isinstance(
            weight_path, str
        ), f"cfg.train.load_weight:{self.cfg.train.load_weight} must be string"
        if resume:
            assert self.checkpointer.has_checkpoint()
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
            assert self.start_iter == (
                self.checkpointer.resume_or_load(None, resume=True).get("iter", -1) + 1
            )
        elif len(weight_path) != 0:
            assert os.path.isdir(
                weight_path
            ), f"cfg.train.load_weight:{self.cfg.train.load_weight} must be directory"
            self.checkpointer.load(weight_path, checkpointables=[])

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),  # for beauty lr scheduler printer in `nn.Graph` mode
            hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.train.checkpointer.period),
        ]

        if self.cfg.train.evaluation.enabled:
            assert self.cfg.train.evaluation.eval_iter > 0, "run_iter must be positive number"

            def test_and_save_results():
                model = self.graph_eval if self.cfg.graph.enabled else self.model
                self._last_eval_results = self.test(self.cfg, self.test_loader, model)
                return self._last_eval_results

            ret.append(hooks.EvalHook(self.cfg.train.evaluation.eval_period, test_and_save_results))
            ret.append(
                hooks.BestCheckpointer(
                    self.cfg.train.evaluation.eval_period,
                    self.checkpointer,
                    val_metric=try_get_key(
                        self.cfg, "train.evaluation.eval_metric", default="Acc@1"
                    ),
                    mode=try_get_key(self.cfg, "train.evaluation.eval_mode", default="max"),
                )
            )

        if dist.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), self.cfg.train.log_period))
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
                JSONWriter(os.path.join(self.cfg.train.output_dir, "metrics.json")),
                TensorboardXWriter(self.cfg.train.output_dir),
            ]
        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.global_batch_size, self.max_iter),
            JSONWriter(os.path.join(self.cfg.train.output_dir, "metrics.json")),
            TensorboardXWriter(self.cfg.train.output_dir),
        ]

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step(self.get_batch, self.cfg.train.input_placement_device)

    def np_to_global(np):
        t = flow.from_numpy(np)
        return t.to_global(
            placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0), check_meta=False
        )

    @classmethod
    def get_batch(
        cls,
        data,
        input_placement_device: str = "cuda",
        mixup_func: Optional[Callable] = None,
    ):
        """
        Convert batched local tensor to distributed tensor for model step running.
        """
        ret_dict = {}
        for key, value in data.items():
            ret_dict[key] = cls.np_to_global(value)
        return ret_dict

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            flow.nn.Module:

        It now calls :func:`libai.models.build_model`.
        Overwrite it if you'd like a different model.
        """
        assert try_get_key(cfg, "model") is not None, "cfg must contain `model` namespace"
        # Set model fp16 option because of embedding layer `white_identity` manual
        # insert for amp training if provided.
        if try_get_key(cfg.model, "cfg.amp_enabled") is not None:
            cfg.model.cfg.amp_enabled = cfg.train.amp.enabled and cfg.graph.enabled
        # In case some model define without cfg keyword.
        elif try_get_key(cfg.model, "amp_enabled") is not None:
            cfg.model.amp_enabled = cfg.train.amp.enabled and cfg.graph.enabled
        model = build_model(cfg.model)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        model._apply(dist.convert_to_distributed_default_setting)
        return model

    @classmethod
    def build_graph(cls, cfg, model, loss=None, optimizer=None, lr_scheduler=None, is_train=True):
        assert try_get_key(cfg, "graph") is not None, "cfg must contain `graph` namespace"
        graph = build_ctr_graph(cfg, model, loss, optimizer, lr_scheduler, is_train)
        debug_graph = try_get_key(cfg, "graph.debug", default=-1)
        if debug_graph >= 0:
            logger = logging.getLogger(__name__)
            logger.info("Graph debug mode on, automatically output debug info.")
            graph.debug(cfg.graph.debug)
        return graph

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            flow.optim.Optimizer:

        It now calls :func:`libai.optim.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        assert try_get_key(cfg, "optim") is not None, "cfg must contain `optim` namespace"
        return build_optimizer(cfg.optim, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`libai.scheduler.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        assert (
            try_get_key(cfg, "train.scheduler") is not None
        ), "cfg.train must contain `scheduler` namespace"
        return build_lr_scheduler(cfg.train.scheduler, optimizer)

    @classmethod
    def build_data_loader(cls, cfg):
        """
        Returns:
            iterable

        Overwrite it if you'd like a different data loader.
        """
        assert (
            try_get_key(cfg, "dataloader.train") is not None
        ), "cfg must contain `dataloader.train` namespace"
        logger = logging.getLogger(__name__)
        logger.info("Prepare training, validating, testing set")
        #todo: cfg.dataloader.train.seed = cfg.train.seed

        train_loader = instantiate(cfg.dataloader.train, _recursive_=False)
        valid_loader = instantiate(cfg.dataloader.validation, _recursive_=False)
        test_loader = instantiate(cfg.dataloader.test, _recursive_=False)
        return train_loader, valid_loader, test_loader

    @classmethod
    def build_evaluator(cls, cfg):
        evaluator = instantiate(cfg.train.evaluation.evaluator)
        return evaluator

    @classmethod
    def test(cls, cfg, test_loaders, model, evaluator=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            test_loaders: list [dataloader1, dataloader2, ...]
            model (nn.Graph):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        # TODO: support multi evaluator
        # if isinstance(evaluators, DatasetEvaluator):
        #     evaluators = [evaluators]
        test_batch_size = cfg.train.test_micro_batch_size * dist.get_data_parallel_size()
        evaluator = cls.build_evaluator(cfg) if not evaluator else evaluator

        results = OrderedDict()
        for idx, data_loader in enumerate(test_loaders):
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            dataset_name = type(data_loader.dataset).__name__
            # TODO: support multi evaluator
            # if evaluators is not None:
            #     evaluator = evaluators[idx]
            # else:
            #     try:
            #         evaluator = cls.build_evaluator(cfg)
            #     except NotImplementedError:
            #         logger.warn(
            #             "No evaluator found. Use `CTRTrainer.test(evaluators=)`, "
            #             "or implement its `build_evaluator` method."
            #         )
            #         results[dataset_name] = {}
            #         continue
            results_i = inference_on_dataset(
                model,
                data_loader,
                test_batch_size,
                cfg.train.evaluation.eval_iter,
                cls.get_batch,
                cfg.train.input_placement_device,
                evaluator,
            )
            results[dataset_name] = results_i
            if dist.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(
                        colored(dataset_name, "green")
                    )
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
