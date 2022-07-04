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
from libai.models import build_graph, build_model
from libai.optim import build_optimizer
from libai.scheduler import build_lr_scheduler
from libai.tokenizer import build_tokenizer
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
    # Set total training samples.
    cfg.train.samples = cfg.train.train_iter * cfg.train.global_batch_size


def _compile_dependencies():
    logger = logging.getLogger(__name__)
    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if dist.get_local_rank() == 0:
        start_time = time.time()
        logger.info("> compiling dataset index builder ...")
        from libai.data.data_utils import compile_helper

        compile_helper()
        logger.info(
            ">>> done with dataset index builder. Compilation time: {:.3f} "
            "seconds".format(time.time() - start_time)
        )

    dist.synchronize()
    if dist.get_local_rank() == 0:
        logger.info(
            ">>> done with compiling. "
            "Compilation time: {:.3f} seconds".format(time.time() - start_time)
        )


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the libai logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Setup the distributed environment
    4. Setup tokenizer if it's an NLP related task
    5. Check batch_size
    6. Backup the config to the output directory
    7. Compile dependencies

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
                args.config_file,
                _highlight(open(args.config_file, "r").read(), args.config_file),
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

    _compile_dependencies()


class DefaultTrainer(TrainerBase):
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

        trainer = DefaultTrainer(cfg)
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

        # Initialize tokenizer
        self.tokenizer = self.build_tokenizer(cfg)

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
        if cfg.graph.enabled:
            cfg.dataloader.consumed_samples = self.start_iter * cfg.train.global_batch_size
        else:
            cfg.dataloader.consumed_samples = (
                self.start_iter * cfg.train.global_batch_size // cfg.train.num_accumulation_steps
            )

        self.train_loader = None
        self.test_loader = []

        train_loader, val_loader, test_loader = self.build_train_loader(cfg, self.tokenizer)
        self.train_loader = train_loader

        if val_loader is not None:
            self.test_loader.append(val_loader)
        if test_loader is not None:
            self.test_loader.append(test_loader)

        self.test_loader.extend(self.build_test_loader(cfg, self.tokenizer))

        if cfg.train.rdma_enabled:
            # set rdma
            flow.env.init_rdma()

        # Automatically scale the hyperparams
        self.auto_scale_hyperparams(cfg, self.train_loader)

        # Assume these objects must be constructed in this order.
        self.model = self.build_model(cfg)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.lr_scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        if cfg.graph.enabled:
            self.graph_train = self.build_graph(
                cfg, self.model, self.optimizer, self.lr_scheduler, is_train=True
            )
            self.graph_eval = self.build_graph(cfg, self.model, is_train=False)
            self._trainer = GraphTrainer(
                self.graph_train, self.train_loader, cfg.train.num_accumulation_steps
            )
        else:
            self._trainer = EagerTrainer(
                self.model, self.train_loader, self.optimizer, cfg.train.num_accumulation_steps
            )

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
        self.resume_or_load(cfg.train.resume)
        cfg.train.start_iter = self.start_iter

        # global_batch_size = micro_batch_size * num_gpus * num_accumulation_steps
        # When using gradient accumulation in graph mode, each run_step
        # handle `global_batch_size` samples.
        # When using gradient accumulation in eager mode, each run_step just handle
        # `micro_batch_size * num_gpus` samples, so we need to divide `num_accumulation_steps`
        # to get the actual `batch_size` for computing `throughput` and `consumed_samples`
        self.global_batch_size = (
            cfg.train.global_batch_size
            if cfg.graph.enabled
            else cfg.train.global_batch_size // cfg.train.num_accumulation_steps
        )
        self.max_iter = cfg.train.train_iter

        self.register_hooks(self.build_hooks())

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

    @classmethod
    def get_batch(
        cls,
        data: Instance,
        input_placement_device: str = "cuda",
        mixup_func: Optional[Callable] = None,
    ):
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
            value.to_global(device_type=input_placement_device)
            ret_dict[key] = value.tensor
        return ret_dict

    @classmethod
    def build_tokenizer(cls, cfg):
        """
        Returns:
            libai.tokenizer.PreTrainedTokenizer:

        It now calls :func:`libai.tokenizer.build_tokenizer`.
        """
        tokenizer = None
        if try_get_key(cfg, "tokenization") is not None:
            tokenizer = build_tokenizer(cfg.tokenization)
            # FIXME(lxy): In case model is not defined with cfg, the `vocab_size` can be
            # accessed by `model.vocab_size`.
            if try_get_key(cfg, "model.cfg.vocab_size", default=None) is not None:
                # In case the model does not need vocab_size as argument
                multiple = (
                    cfg.tokenization.make_vocab_size_divisible_by
                    * cfg.train.dist.tensor_parallel_size
                )
                cfg.model.cfg.vocab_size = tokenizer.padded_vocab_size(multiple)
        return tokenizer

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
        model.apply(dist.convert_to_distributed_default_setting)
        return model

    @classmethod
    def build_graph(cls, cfg, model, optimizer=None, lr_scheduler=None, is_train=True):
        assert try_get_key(cfg, "graph") is not None, "cfg must contain `graph` namespace"
        graph = build_graph(cfg, model, optimizer, lr_scheduler, is_train)
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
        assert (
            try_get_key(cfg, "train.scheduler") is not None
        ), "cfg.train must contain `scheduler` namespace"
        return build_lr_scheduler(cfg.train.scheduler, optimizer)

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
        if cfg.graph.enabled:
            # In static graph mode, data will be sliced in nn.Graph automatically,
            # dataloader will get micro-batch-size and data will be concated
            # in graph_trainer.run_step to get mini-batch-size.
            cfg.dataloader.train.train_batch_size = cfg.train.train_micro_batch_size
        else:
            # In eager mode, gradient accumulation will act like PyTorch, so dataloader
            # will get micro-batch-size
            cfg.dataloader.train.train_batch_size = cfg.train.train_micro_batch_size
        cfg.dataloader.train.test_batch_size = cfg.train.test_micro_batch_size
        cfg.dataloader.train.seed = cfg.train.seed

        if hasattr(cfg.dataloader.train, "train_val_test_num_samples"):
            eval_iter = (
                (cfg.train.train_iter // cfg.train.evaluation.eval_period + 1)
                * cfg.train.evaluation.eval_iter
                if cfg.train.evaluation.enabled
                # samples for test_dataset must be larger than 0 even if there is no evaluation
                else 1
            )
            test_iter = cfg.train.evaluation.eval_iter if cfg.train.evaluation.enabled else 1

            cfg.dataloader.train.train_val_test_num_samples = [
                int(cfg.train.samples),
                int(eval_iter * cfg.train.test_micro_batch_size * dist.get_data_parallel_size()),
                int(test_iter * cfg.train.test_micro_batch_size * dist.get_data_parallel_size()),
            ]
        if OmegaConf.is_list(cfg.dataloader.train.dataset):
            for dataset in cfg.dataloader.train.dataset:
                if hasattr(dataset, "seed"):
                    dataset.seed = cfg.train.seed
        else:
            dataset = cfg.dataloader.train.dataset
            if hasattr(dataset, "seed"):
                dataset.seed = cfg.train.seed

        # Set tokenizer for each dataset
        if tokenizer:
            if OmegaConf.is_list(cfg.dataloader.train.dataset):
                for dataset in cfg.dataloader.train.dataset:
                    dataset.tokenizer = tokenizer
            else:
                cfg.dataloader.train.dataset.tokenizer = tokenizer

        train_loader, valid_loader, test_loader = instantiate(
            cfg.dataloader.train, _recursive_=False
        )
        return train_loader, valid_loader, test_loader

    @classmethod
    def build_test_loader(cls, cfg, tokenizer=None):
        """
        Returns:
            iterable

        It now calls :func:`libai.data.build_image_test_loader` for CV tasks
        or :func:`libai.data.build_nlp_test_loader` for NLP tasks.
        Overwrite it if you'd like a different data loader.
        """
        # If there is no test_loader, just return []
        if not try_get_key(cfg, "dataloader.test", default=False):
            return []
        logger = logging.getLogger(__name__)
        logger.info("Prepare testing set")
        assert OmegaConf.is_list(
            cfg.dataloader.test
        ), f"dataloader.test must be list but got type of {type(cfg.dataloader.test)}"
        for i in range(len(cfg.dataloader.test)):
            cfg.dataloader.test[i].test_batch_size = cfg.train.test_micro_batch_size
            cfg.dataloader.test[i].seed = cfg.train.seed  # set seed
            if tokenizer:
                cfg.dataloader.test[i].dataset.tokenizer = tokenizer
        # list[dataloader1, dataloader2, ...]
        test_loader = instantiate(cfg.dataloader.test, _recursive_=False)
        return test_loader

    @classmethod
    def auto_scale_hyperparams(cls, cfg, data_loader):
        logger = logging.getLogger(__name__)
        log_info = ""

        # Get or set default iteration cfg
        train_iter = try_get_key(cfg, "train.train_iter", default=0)
        train_epoch = try_get_key(cfg, "train.train_epoch", default=0)
        warmup_ratio = try_get_key(cfg, "train.warmup_ratio", default=0)
        assert (
            warmup_ratio < 1 and warmup_ratio >= 0
        ), "warmup_ratio must be in [0, 1) that presents the ratio of warmup iter to the train iter"

        # Automatically scale iteration num depend on the settings
        # The total iters in one epoch is `len(dataset) / global_batch_size`
        cfg.train.train_iter = max(
            math.ceil(len(data_loader.dataset) * train_epoch / cfg.train.global_batch_size),
            train_iter,
        )
        cfg.train.warmup_iter = math.ceil(cfg.train.train_iter * cfg.train.warmup_ratio)
        if not cfg.graph.enabled:
            # In eager mode, dataloader only get micro-batch-size each iter,
            # which is mini-batch-size // num_accumulation, so scale `train_iter`
            # and `warmup_iter` to be consistent with static graph mode.
            cfg.train.train_iter *= cfg.train.num_accumulation_steps
            cfg.train.warmup_iter *= cfg.train.num_accumulation_steps
        log_info += "Auto-scaling the config to train.train_iter={}, train.warmup_iter={}".format(
            cfg.train.train_iter, cfg.train.warmup_iter
        )

        # Automatically scale the milestones
        if try_get_key(cfg, "train.scheduler.milestones"):
            if len(
                [
                    milestone
                    for milestone in cfg.train.scheduler.milestones
                    if milestone < 0 or milestone >= 1
                ]
            ):
                raise ValueError(
                    "milestones should be a list of increasing ratio in [0, 1), but got {}".format(
                        cfg.train.scheduler.milestones
                    )
                )
            cfg.train.scheduler.milestones = [
                int(milestone * cfg.train.train_iter)
                for milestone in cfg.train.scheduler.milestones
            ]
            log_info += f", scheduler milestones={cfg.train.scheduler.milestones}"
        logger.info(log_info)

        # Global scheduler cfg
        cfg.train.scheduler.warmup_iter = cfg.train.warmup_iter
        cfg.train.scheduler.max_iter = cfg.train.train_iter

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
            #             "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
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
