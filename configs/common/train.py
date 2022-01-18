# fmt: off
train = dict(
    output_dir="./demo_output/test_config",

    micro_batch_size=32,
    global_batch_size=None,
    num_accumulation_steps=None,

    start_iter=0,
    train_iter=10000,
    warmup_epoch=None,
    train_epoch=None,
    lr_decay_iter=None,
    eval_iter=10000,
    lr_warmup_fraction=0.01,
    amp=dict(enabled=False),  # options for Automatic Mixed Precision
    checkpointer=dict(period=5000, max_to_keep=100),  # options for PeriodicCheckpointer
    load_weight="",
    eval_period=5000,
    log_period=20,
    consumed_train_samples=0,
    consumed_valid_samples=0,
    train_samples=None,

    # Distributed arguments
    dist=dict(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    ),
    seed=1234,
    # NCCL fusion threshold megabytes, set to 0 to compatible with previous version of OneFlow
    nccl_fusion_threshold_mb=16,
    # Maximum number of ops of NCCL fusion, set to 0 to compatible with previous version of OneFlow
    nccl_fusion_max_ops=24,
    enable_use_compute_stream=True,
)
# fmt: on
