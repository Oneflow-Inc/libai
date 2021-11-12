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

import math
import argparse
from .utils import print_rank_0

from core.data import DATASETS
from core.models import MODELS
from core.criterion import CRITERIONS


def parse_args(ignore_unknown_args=True):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description="Arguments", allow_abbrev=False)

    _add_data_args(parser)
    _add_training_args(parser)
    _add_validation_args(parser)
    _add_checkpointing_args(parser)
    _add_mixed_precision_args(parser)
    _add_distributed_args(parser)
    _add_misc_args(parser)
    _add_model_args(parser)
    _add_criterion_args(parser)

    args, _ = parser.parse_known_args()
    
    if hasattr(args, "data_type"):
        DATASETS[args.data_type].add_args(parser)

    if hasattr(args, "model_type"):
        MODELS[args.model_type].add_args(parser)
    
    if hasattr(args, "criterion"):
        CRITERIONS[args.criterion].add_args(parser)
    
    args = parser.parse_args()

    _check_model_size(args)
    _check_parallel_size(args)
    _check_batch_size(args)
    _check_train_iters(args)
    _check_lr_decay_and_warmup(args)

    args.padded_vocab_size = _pad_vocab_size(
        args.vocab_size,
        args.make_vocab_size_divisible_by,
        args.tensor_model_parallel_size,
    )

    _print_args(args)
    return args


def _str_list(x):
    return x.split(",")


def _int_list(x):
    return list(map(int, x.split(",")))


def _float_list(x):
    return list(map(float, x.split(",")))


def _str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def _check_model_size(args):
    if args.hidden_size % args.num_attention_heads != 0:
        raise ValueError(
            f"hidden size {args.hidden_size} must be divisible by"
            f" number of attention heads {args.num_attention_heads}"
        )

    if args.num_attention_heads % args.tensor_model_parallel_size != 0:
        raise ValueError(
            f"number of attention heads {args.num_attention_heads} must be divisible by"
            f" tensor model parallel size {args.tensor_model_parallel_size}"
        )

    if args.num_layers % args.pipeline_model_parallel_size != 0:
        raise ValueError(
            f"number of layers {args.num_layers} must be divisible by"
            f" pipeline model parallel size {args.pipeline_model_parallel_size}"
        )


def _check_parallel_size(args):
    world_size = args.num_gpus_per_node * args.num_nodes
    model_parallel_size = (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    )
    if world_size % model_parallel_size != 0:
        raise ValueError(
            f"world_size {world_size} must be divisible by model_parallel_size {model_parallel_size}"
        )

    args.data_parallel_size = world_size // model_parallel_size



def _check_batch_size(args):
    if args.micro_batch_size is not None and args.global_batch_size is not None:
        if args.num_accumulation_steps is None:
            if (
                args.global_batch_size
                % (args.micro_batch_size * args.data_parallel_size)
                != 0
            ):
                raise ValueError(
                    f"global_batch_size {args.global_batch_size} must be divisible by "
                    f"micro_batch_size * data_parallel_size ({args.micro_batch_size} * {args.data_parallel_size})"
                )

            args.num_accumulation_steps = args.global_batch_size // (
                args.micro_batch_size * args.data_parallel_size
            )
        else:
            if (
                args.global_batch_size
                != args.micro_batch_size
                * args.data_parallel_size
                * args.num_accumulation_steps
            ):
                raise ValueError(
                    f"global_batch_size {args.global_batch_size} must equal"
                    " micro_batch_size * data_parallel_size * num_accumulation_steps"
                    f" ({args.micro_batch_size} * {args.data_parallel_size} * {args.num_accumulation_steps})"
                )
    elif args.micro_batch_size is not None and args.global_batch_size is None:
        if args.num_accumulation_steps is None:
            args.num_accumulation_steps = 1

        args.global_batch_size = (
            args.micro_batch_size
            * args.data_parallel_size
            * args.num_accumulation_steps
        )
    elif args.micro_batch_size is None and args.global_batch_size is not None:
        if args.num_accumulation_steps is None:
            args.num_accumulation_steps = 1

        if (
            args.global_batch_size
            % (args.data_parallel_size * args.num_accumulation_steps)
            != 0
        ):
            raise ValueError(
                f"global_batch_size {args.global_batch_size} must be divisible by "
                "data_parallel_size * num_accumulation_steps "
                f"({args.data_parallel_size} * {args.num_accumulation_steps})"
            )

        args.micro_batch_size = args.global_batch_size // (
            args.data_parallel_size * args.num_accumulation_steps
        )
    else:
        raise ValueError("micro_batch_size and global_batch_size must be set either")

    assert args.num_accumulation_steps is not None
    if args.num_accumulation_steps > 1 and args.use_external_dataset:
        raise ValueError(
            "num_accumulation_steps couldn't be greater than 1 when use external dataset"
        )


def _check_train_iters(args):
    if args.train_iters is None and args.train_samples is None:
        raise ValueError("train_iters and train_samples must be set either")

    if args.train_iters is None:
        if args.train_samples % args.global_batch_size != 0:
            raise ValueError(
                f"train_samples {args.train_samples} must be divisible by "
                f"global_batch_size {args.global_batch_size}"
            )

        args.train_iters = args.train_samples // args.global_batch_size

    if args.train_samples is None:
        args.train_samples = args.train_iters * args.global_batch_size


def _check_lr_decay_and_warmup(args):
    if args.lr_decay_style == "cosine" and args.lr_decay_iters is None:
        raise ValueError(
            f"lr_decay_iters must be set when lr_decay_style is {args.lr_decay_style}"
        )

    if (
        args.lr_warmup_iters is None
        and args.lr_warmup_fraction is not None
        and args.lr_decay_iters is not None
    ):
        args.lr_warmup_iters = int(args.lr_warmup_fraction * args.lr_decay_iters)


def _pad_vocab_size(vocab_size, alignment, tensor_model_parallel_size):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""
    assert isinstance(alignment, int)
    if alignment == 0:
        return vocab_size

    alignment *= tensor_model_parallel_size

    padded_vocab_size = int(math.ceil(vocab_size / alignment)) * alignment
    print_rank_0(
        " > padded vocab (size: {}) with {} dummy tokens "
        "(new size: {})".format(
            vocab_size, padded_vocab_size - vocab_size, padded_vocab_size
        )
    )
    return padded_vocab_size


def _print_args(args):
    """Print arguments."""
    print_rank_0(
        "------------------------ arguments ------------------------", flush=True
    )
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print_rank_0(arg, flush=True)
    print_rank_0(
        "-------------------- end of arguments ---------------------", flush=True
    )


def _add_data_args(parser):
    group = parser.add_argument_group(title="data and dataloader")
    print(DATASETS.keys())
    group.add_argument("--data-type", required=True, choices=DATASETS.keys(), help="Dataset type.")

    group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path join the data index file and binary file prefix.",
    )
    group.add_argument(
        "--split",
        type=_int_list,
        default=[969, 30, 1],
        help="Comma-separated list of proportions for training,"
        " validation, and test split. For example the split "
        "`90,5,5` will use 90%% of data for training, 5%% for "
        "validation and 5%% for test.",
    )
    group.add_argument(
        "--seed", type=int, default=12345, help="Random seed used for data gen."
    )
    group.add_argument("--vocab-size", type=int, default=50257, help="Vocab size.")
    group.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length to process.",
    )
    group.add_argument(
        "--use-external-dataset",
        action="store_true",
        help="Use external megatron dataset.",
    )
    group.add_argument(
        "--make-vocab-size-divisible-by",
        type=int,
        default=1,
        help="make vocabulary size devisible"
    )

    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title="training")

    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Batch size per model instance (local batch size). "
        "Global batch size is local batch size times data "
        "parallel size times number of micro batches.",
    )
    group.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Training batch size. If set, it should be a "
        "multiple of micro-batch-size times data-parallel-size. "
        "If this value is None, then "
        "use micro-batch-size * data-parallel-size as the "
        "global batch size. This choice will result in 1 for "
        "number of micro-batches.",
    )
    parser.add_argument(
        "--num-accumulation-steps",
        type=int,
        default=None,
        help="Number of accumulation micro steps before gradient update, "
        "Global batch size = num_accumulation_steps * batch_size",
    )
    group.add_argument(
        "--checkpoint-activations",
        action="store_true",
        dest="checkpoint_activations",
        help="Checkpoint activation to allow for training "
        "with larger models, sequences, and batch sizes.",
    )
    group.add_argument(
        "--train-iters",
        type=int,
        default=None,
        help="Total number of iterations to train over all "
        "training runs. Note that either train-iters or "
        "train-samples should be provided.",
    )
    group.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Total number of samples to train over all "
        "training runs. Note that either train-iters or "
        "train-samples should be provided.",
    )
    group.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "adam", "adamw"],
        help="Optimizer. <sgd|adam|adamw>.",
    )
    group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate. Depending on decay style "
        "and initial warmup, the learing rate at each "
        "iteration would be different.",
        dest="lr",
    )
    group.add_argument(
        "--lr-decay-style",
        type=str,
        default="cosine",
        choices=["none", "cosine"],
        help="Learning rate decay function.",
    )
    group.add_argument(
        "--lr-decay-iters",
        type=int,
        default=None,
        help="number of iterations to decay learning rate over,"
        " If None defaults to `--train-iters`",
    )
    group.add_argument(
        "--lr-warmup-fraction",
        type=float,
        default=None,
        help="fraction of lr-warmup-(iters/samples) to use " "for warmup (as a float)",
    )
    group.add_argument(
        "--lr-warmup-iters",
        type=int,
        default=None,
        help="number of iterations to linearly warmup " "learning rate over.",
    )
    group.add_argument(
        "--weight-decay",
        type=float,
        default=0.,
        help="Weight decay coefficient for L2 regularization."
    )
    group.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Gradient clipping based on global L2 norm.",
    )
    group.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Minimum value for learning rate. The scheduler"
        "clip values below this threshold.",
    )
    group.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="First coefficient for computing running averages of"
        "gradient and its square",
    )
    group.add_argument(
        "--adam-beta2",
        type=float,
        default=0.999,
        help="Second coefficient for computing running averages of"
        "gradient and its square",
    )
    group.add_argument(
        "--adam-eps",
        type=float,
        default=1e-08,
        help="Term added to the denominator to improve" "numerical stability",
    )
    group.add_argument("--log", type=str, default="./output", help="log directory")
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        dest="log_interval",
        help="Report loss and timing interval.",
    )

    return parser


def _add_validation_args(parser):
    # group = parser.add_argument_group(title="validation")
    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title="checkpointing")

    group.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Output directory to save checkpoints to.",
    )
    group.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Number of iterations between checkpoint saves.",
    )
    group.add_argument(
        "--load",
        type=str,
        default=None,
        dest="checkpoint_load_path",
        help="Directory containing a model checkpoint.",
    )
    group.add_argument(
        "--save-last",
        action="store_true",
        default=False,
        help="save model snapshot for last iteration",
    )
    group.add_argument(
        "--save-init",
        action="store_true",
        default=False,
        help="save model snapshot for inited",
    )

    return parser


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title="mixed precision")

    group.add_argument("--fp16", action="store_true", help="Run model in fp16 mode.")
    group.add_argument(
        "--loss-scale",
        type=float,
        default=None,
        help="Static loss scaling, positive power of 2 "
        "values can improve fp16 convergence. If None, dynamic"
        "loss scaling is used.",
    )
    group.add_argument(
        "--initial-loss-scale",
        type=float,
        default=None,
        help="Initial loss-scale for dynamic loss scaling.",
    )
    group.add_argument(
        "--loss-scale-window",
        type=float,
        default=1000,
        help="Window over which to raise/lower dynamic scale.",
    )

    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title="distributed")

    group.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=1,
        help="Degree of tensor model parallelism.",
    )
    group.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        default=1,
        help="Degree of pipeline model parallelism.",
    )
    group.add_argument(
        "--num-gpus-per-node",
        type=int,
        default=1,
        help="Number of gpu devices per node/machine.",
    )
    group.add_argument(
        "--num-nodes", type=int, default=1, help="Node/Machine number for training."
    )
    return parser


def _add_misc_args(parser):
    group = parser.add_argument_group(title="misc")
    group.add_argument(
        "--graph", action="store_true", help="Use graph mode.",
    )
    return parser


def _add_model_args(parser):
    group = parser.add_argument_group(title="model")
    group.add_argument("--model-type", required=True, choices=MODELS.keys(), help="Model architecture.")
    return parser

def _add_criterion_args(parser):
    group = parser.add_argument_group(title="criterion")
    group.add_argument("--criterion", default="cross_entropy", choices=CRITERIONS.keys(), help="Criterion.")
    return parser

