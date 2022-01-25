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
from collections import OrderedDict

import omegaconf
import oneflow as flow
from termcolor import colored

from libai.config import LazyConfig, try_get_key
from libai.config.instantiate import instantiate
from libai.data import Instance
from libai.evaluation import ClsEvaluator, inference_on_dataset, print_csv_format
from libai.models import build_graph, build_model
from libai.optim import build_optimizer
from libai.scheduler import build_lr_scheduler
from libai.tokenizer import build_tokenizer
from libai.trainer import hooks
from libai.trainer.trainer import EagerTrainer, GraphTrainer, TrainerBase
from libai.utils import distributed as dist
from libai.utils.checkpoint import Checkpointer
from libai.utils.events import CommonMetricPrinter, JSONWriter
from libai.utils.logger import setup_logger


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


def _check_batch_size(cfg):
    train_micro_batch_size = try_get_key(cfg, "train.train_micro_batch_size", default=None)
    global_batch_size = try_get_key(cfg, "train.global_batch_size", default=None)
    num_accumulation_steps = try_get_key(cfg, "train.num_accumulation_steps", default=None)

    if train_micro_batch_size is not None and global_batch_size is not None:
        if num_accumulation_steps is None:
            if global_batch_size % (train_micro_batch_size * dist.get_data_parallel_size()) != 0:
                raise ValueError(
                    f"global_batch_size {global_batch_size} must be divisible by "
                    "train_micro_batch_size * data_parallel_size "
                    f"({train_micro_batch_size} * {dist.get_data_parallel_size()})"
                )

            cfg.train.num_accumulation_steps = global_batch_size // (
                train_micro_batch_size * dist.get_data_parallel_size()
            )

        else:
            if (
                global_batch_size
                != train_micro_batch_size * dist.get_data_parallel_size() * num_accumulation_steps
            ):
                raise ValueError(
                    f"global_batch_size {global_batch_size} must equal to "
                    "train_micro_batch_size * data_parallel_size * num_accumulation_steps "
                    f"({train_micro_batch_size} * {dist.get_data_parallel_size()} * {num_accumulation_steps})"  # noqa
                )
    elif train_micro_batch_size is not None and global_batch_size is None:
        if num_accumulation_steps is None:
            cfg.train.num_accumulation_steps = 1

        cfg.train.global_batch_size = (
            train_micro_batch_size
            * dist.get_data_parallel_size()
            * cfg.train.num_accumulation_steps
        )
    elif train_micro_batch_size is None and global_batch_size is not None:
        if num_accumulation_steps is None:
            cfg.train.num_accumulation_steps = 1

        if (
            global_batch_size % (dist.get_data_parallel_size() * cfg.train.num_accumulation_steps)
            != 0
        ):
            raise ValueError(
                f"global_batch_size {global_batch_size} must be divisible by "
                "data_parallel_size * num_accumulation_steps "
                f"({dist.get_data_parallel_size()} * {cfg.train.num_accumulation_steps})"
            )

        cfg.train.train_micro_batch_size = global_batch_size // (
            dist.get_data_parallel_size() * cfg.train.num_accumulation_steps
        )
    else:
        raise ValueError("train_micro_batch_size and global_batch_size must be set either")


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:
    1. Set up the libai logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Setup the distributed environment
    4. Setup tokenizer if it's NLP related task
    5. Check batch_size
    6. Backup the config to the output directory
    Args:
        args (argparse.NameSpace): the command line arguments to be logged
    """

    output_dir = try_get_key(cfg, "train.output_dir")
    if dist.is_main_process() and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cfg.train.resume = args.resume

    rank = dist.get_rank()
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, dist.get_world_size()))
    logger.info("Command line arguments: " + str(args))

    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, _highlight(open(args.config_file, "r").read(), args.config_file),
            )
        )

    dist.setup_dist_util(cfg.train.dist)

    _check_batch_size(cfg)

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
    flow.boxing.nccl.enable_use_compute_stream(
        try_get_key(cfg, "train.enable_use_compute_stream", default=True)
    )


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

        # setup_logger is not called for LiBai
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()

        # Initialize tokenizer
        self.tokenizer = None
        if try_get_key(cfg, "tokenization.setup", default=False):
            self.tokenizer = self.build_tokenizer(cfg)

        # Assume these objects must be constructed in this order.
        self.model = self.build_model(cfg)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.lr_scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
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
        self.resume_or_load(cfg.train.resume)
        cfg.train.start_iter = self.start_iter

        self.train_loader = None
        self.test_loader = []

        train_loader, val_loader, test_loader = self.build_train_loader(cfg, self.tokenizer)
        self.train_loader = train_loader

        if val_loader is not None:
            self.test_loader.append(val_loader)
        if test_loader is not None:
            self.test_loader.append(test_loader)

        self.test_loader.extend(self.build_test_loader(cfg))

        if cfg.graph.enabled:
            self.graph_train = self.build_graph(
                cfg, self.model, self.optimizer, self.lr_scheduler, is_train=True
            )
            self.graph_eval = self.build_graph(cfg, self.model, is_train=False)
            self._trainer = GraphTrainer(self.graph_train, self.train_loader)
        else:
            self._trainer = EagerTrainer(self.model, self.train_loader, self.optimizer)

        self.global_batch_size = cfg.train.global_batch_size
        self.max_iter = cfg.train.train_iter

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.train.output_dir` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.train.load_weight`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.train.load_weight` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        if resume:
            if self.checkpointer.has_checkpoint():
                # The checkpoint stores the training iteration that just finished, thus we start
                # at the next iteration (or iter zero if there's no checkpoint).
                self.start_iter = (
                    self.checkpointer.resume_or_load(None, resume=True).get("iter", -1) + 1
                )
            else:
                # This is considered as an independent training.
                self.checkpointer.load(self.cfg.train.load_weight, checkpointables=[])
        else:
            self.start_iter = 0

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
            hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.train.checkpointer.period),
        ]

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.test_loader, self.graph_eval)
            return self._last_eval_results

        ret.append(hooks.EvalHook(self.cfg.train.eval_period, test_and_save_results))

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
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]
        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.global_batch_size, self.max_iter),
            JSONWriter(os.path.join(self.cfg.train.output_dir, "metrics.json")),
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
        self._trainer.run_step(self.get_batch)

    @classmethod
    def get_batch(cls, data: Instance):
        """
        Convert batched local tensor to distributed tensor for model step running.

        If you want to do something with batched data before model, (e.g. mixup),
        you can rewrite this function.
        """
        if isinstance(data, flow.utils.data._utils.worker.ExceptionWrapper):
            data.reraise()

        ret_dict = {}
        ret_list = []
        for key, value in data.get_fields().items():
            value.to_consistent()
            ret_dict[key] = value.tensor
            ret_list.append(value.tensor)
        # FIXME(l1aoxingyu): `nn.Graph` cannot accpet key-value arguments right now,
        # just pass list instead.
        return ret_list

    @classmethod
    def build_tokenizer(cls, cfg):
        assert (
            try_get_key(cfg, "tokenization") is not None
        ), "cfg must contain `tokenization` namespace"
        return build_tokenizer(cfg)

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            flow.nn.Module:
        It now calls :func:`libai.models.build_model`.
        Overwrite it if you'd like a different model.
        """
        assert try_get_key(cfg, "model") is not None, "cfg must contain `model` namespace"
        model = build_model(cfg.model)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        model.apply(dist.convert_to_distributed_default_setting)
        return model

    @classmethod
    def build_graph(cls, cfg, model, optimizer=None, lr_scheduler=None, is_train=True):
        assert try_get_key(cfg, "graph") is not None, "cfg must contain `graph` namespace"
        graph = build_graph(cfg.graph, model, optimizer, lr_scheduler, is_train)
        logger = logging.getLogger(__name__)
        debug_graph = try_get_key(cfg, "graph.debug", default=-1)
        if debug_graph >= 0:
            logger.info("Graph debug mode on, automatically output debug info.")
            graph.debug(cfg.graph.debug)
        return graph

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
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
        assert try_get_key(cfg, "scheduler") is not None, "cfg must contain `scheduler` namespace"
        return build_lr_scheduler(cfg.scheduler, optimizer)

    @classmethod
    def build_train_loader(cls, cfg, tokenizer=None):
        """
        Returns:
            iterable
        It now calls :func:`libai.data.build_train_valid_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        assert (
            try_get_key(cfg, "dataloader.train") is not None
        ), "cfg must contain `dataloader.train` namespace"
        logger = logging.getLogger(__name__)
        logger.info("Prepare training, validating, testing set")
        cfg.dataloader.train.train_batch_size = cfg.train.train_micro_batch_size
        cfg.dataloader.train.test_batch_size = cfg.train.test_micro_batch_size

        # Set tokenizer for each dataset
        if tokenizer:
            if isinstance(cfg.dataloader.train.dataset, omegaconf.listconfig.ListConfig):
                for dataset in cfg.dataloader.train.dataset:
                    dataset.tokenizer = tokenizer
            else:
                cfg.dataloader.train.dataset.tokenizer = tokenizer

        train_loader, valid_loader, test_loader = instantiate(cfg.dataloader.train)
        return train_loader, valid_loader, test_loader

    @classmethod
    def build_test_loader(cls, cfg):
        # TODO: add doc string
        # If there is no test_loader, just return []
        if not try_get_key(cfg, "dataloader.test", default=False):
            return []
        logger = logging.getLogger(__name__)
        logger.info("Prepare testing set")
        assert isinstance(
            cfg.dataloader.test, omegaconf.listconfig.ListConfig
        ), f"dataloader.test must be list but got type of {type(cfg.dataloader.test)}"
        for i in range(len(cfg.dataloader.test)):
            cfg.dataloader.test[i].test_batch_size = cfg.train.test_micro_batch_size
        # list[dataloader1, dataloader2, ...]
        test_loader = instantiate(cfg.dataloader.test)
        return test_loader

    @classmethod
    def build_evaluator(cls, cfg):
        return ClsEvaluator(cfg)

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
            dataset_name = getattr(data_loader.dataset, "datasetname", "UndefinedDataset")
            # TODO: support multi evaluator
            # if evaluators is not None:
            #     evaluator = evaluators[idx]
            # else:
            #     try:
            #         evaluator = cls.build_evaluator(cfg)
            #     except NotImplementedError:
            #         logger.warn(
            #             "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
            #             "or implement its `build_evaluator` method."
            #         )
            #         results[dataset_name] = {}
            #         continue

            results_i = inference_on_dataset(
                model, data_loader, test_batch_size, cls.get_batch, evaluator
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
