from omegaconf import DictConfig

from libai.config import LazyCall
from libai.scheduler import WarmupCosineLR
from libai.evaluation import ClsEvaluator

# fmt: off
train = dict(

    # Directory where output files are written
    output_dir="./output",

    # `train_micro_batch_size` is number of samples per batch on each GPU.
    # train_mini_batch_size = train_micro_batch_size * num_accumulation_steps.
    # This is also the number of training samples per step (i.e. per iteration).

    # If we use 8 GPUs for data parallel groups, `train_micro_batch_size = 2` and
    # `num_accumulation_steps = 4`, then each GPU will see 2 samples per batch and
    # 8 samples per iteration.
    # Total 64 samples will be trained per iteration across all GPUs.

    # global_batch_size = micro_batch_size  * num_grad_acc * data_parallel_groups
    train_micro_batch_size=32,
    global_batch_size=None,
    num_accumulation_steps=None,

    # The total training iterations
    train_iter=10000,
    # The total training epochs, will be scaled to training iterations automatically.
    # The actual total training iterations will be calculated by the
    # formula `max(train_iter, train_epoch * iter_per_epoch)`.
    train_epoch=0,
    consumed_train_samples=0,
    consumed_valid_samples=0,
    train_samples=None,

    # Fraction of lr-warmup-iters to use for warmup (as a float)
    warmup_ratio=0,

    # The start iteration, usually needn't set it manually.
    # It can be computed automatically when resuming training.
    start_iter=0,

    # Enable automatic mixed precision for training which does not
    # change model's inference behavior.
    amp=dict(enabled=False),

    # Enable activation checkpointing to allow for training
    # with larger models, sequences, and batch sizes.
    # If enabled, checkpoint the input activations of each transformer layers by default.
    activation_checkpoint=dict(enabled=False),

    # NCCL fusion threshold megabytes, set to 0 to
    # compatible with previous version of OneFlow.
    nccl_fusion_threshold_mb=16,

    # Maximum number of ops of NCCL fusion, set to 0 to
    # compatible with previous version of OneFlow.
    nccl_fusion_max_ops=24,

    # Enable ZeRO Optimization to allow for training with larger models.
    # This optimization will reduce optimizer stages memory consumption
    # as described in ZeRO https://arxiv.org/abs/1910.02054.
    zero_optimization=dict(
        enabled=False,
        stage=1,
    ),

    # Save a model checkpoint after every this number of iterations,
    # and maximum number of checkpoint will be kept.
    checkpointer=dict(period=5000, max_to_keep=100),

    # Options for evaluation

    # `test_micro_batch_size` is number of samples per batch on each GPU for testing.
    # If we use 8 GPUs for data parallel groups and `test_micro_batch_size = 2`, then
    # total 16 samples will be used per iteration across all GPUs.
    test_micro_batch_size=32,

    # Enabled evaluation during training, after every `eval_period` number of iterations
    # will perform the evaluation process.
    # You can set the maximum evaluation iterations to run for validation/test.
    # You can also set a customized evaluator for use.
    evaluation=dict(
        enabled=True,
        # evaluator for calculating top-k acc
        evaluator=LazyCall(ClsEvaluator)(topk=(1, 5)),
        eval_period=5000,
        eval_iter=1e5,  # running steps for validation/test

        # Metrics to be used for best model checkpoint.
        eval_metric="Acc@1",
        eval_mode="max",
    ),

    # Path to a checkpoint file to be loaded to the model for training or evaluation.
    load_weight="",

    # Output log to console after every this number of iterations.
    log_period=20,

    # lr_scheduler arguments
    # See libai/scheduler/lr_scheduler.py for definition.
    scheduler=LazyCall(WarmupCosineLR)(
        # In DefaultTrainer we will automatically set `max_iter`
        # and `warmup_iter` by the given train cfg.
        warmup_factor=0.001,
        alpha=0.01,
        warmup_method="linear",
    ),

    # Distributed arguments
    # See https://libai.readthedocs.io/en/latest/tutorials/Getting%20Started.html for more detail.
    dist=dict(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        # users must set the `pipeline_num_layers` attribute when `pipeline_parallel_size > 1`
        pipeline_num_layers=None,
        # users could customize the number of layers in different stages
        # by setting the `custom_pipeline_stage_id ` attribute which is used for
        # manually balance calculation between stages when running pipeline parallelism
        # e.g. you can set `custom_pipeline_stage_id=[0, 0, 0, 1]`
        # for `pipeline_num_layers=4 and pipeline_parallel_size=2`
        # which means the first 3 layers will be placed on stage0 and
        # the last layer will be placed on stage1
        # NOTE: if it is None, LiBai will automatically set pipeline_stage_id
        # `auto_pipeline_stage_id` and `actual_pipeline_stage_id` will be saved in `config.yaml`
        custom_pipeline_stage_id=None,
    ),

    # the device type of input tensors for model, defaults to "cuda".
    # if you want to accelerate the model training when pipeline_parallel > 1
    # you can set `input_placement_device="cpu"` then call input_tensor.to_global()
    # inside your model.forward() method
    # see `libai/models/bert_model.py` as reference
    input_placement_device="cuda",

    # set to `True` to enable rdma for improving speed of pipeline_parallel
    rdma_enabled=True,

    # Set seed to positive to use a fixed seed. Note that a fixed seed increases
    # reproducibility but does not guarantee fully deterministic behavior.
    # Disabling all parallelism further increases reproducibility.
    seed=1234,
)
# fmt: on

train = DictConfig(train)
