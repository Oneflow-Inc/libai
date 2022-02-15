from libai.config import LazyCall
from libai.scheduler import WarmupCosineLR, WarmupMultiStepLR

# fmt: off
train = dict(
    output_dir="./demo_output/test_config",

    # Batch size and gradient accumulation
    train_micro_batch_size=32,
    test_micro_batch_size=32,
    global_batch_size=None,
    num_accumulation_steps=None,
    start_iter=0,
    train_iter=1000,
    train_epoch=0,  # default train epoch is set to 0
    warmup_ratio=0,  # default warmup ratio is set to 0
    lr_decay_iter=None,
    eval_iter=10000,

    # Performance related
    amp=dict(enabled=False),  # options for Automatic Mixed Precision
    recompute_grad=dict(enabled=False),  # options for recompute gradient
    # NCCL fusion threshold megabytes, set to 0 to compatible with previous version of OneFlow
    nccl_fusion_threshold_mb=16,
    # Maximum number of ops of NCCL fusion, set to 0 to compatible with previous version of OneFlow
    nccl_fusion_max_ops=24,

    # ZeRO Optimization
    zero_optimization=dict(
        enabled=False,
        stage=1,
    ),

    checkpointer=dict(period=5000, max_to_keep=100),  # options for PeriodicCheckpointer

    load_weight="",
    eval_period=15000,
    log_period=1,
    consumed_train_samples=0,
    consumed_valid_samples=0,
    train_samples=None,

    # Scheduler arguments
    scheduler=LazyCall(WarmupMultiStepLR)(
        # in DefaultTrainer we will automatically set max_iter
        # and warmup_iter by the given train cfg.
        milestones=[0.999999, ],
        warmup_factor=0.000,
        # alpha=0.01,
        warmup_method="linear",
    ),

    # Distributed arguments
    dist=dict(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    ),
    seed=1234,
)
