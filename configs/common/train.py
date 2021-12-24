train = dict(
    output_dir="./demo_output/test_config",
    start_iter=0,
    train_iters=1000,
    micro_batch_size=32,
    global_batch_size=64,
    save_interval=1000,
    log_interval=20,
    init_checkpoint="",
    max_iter=5,
    amp=dict(enabled=False),  # options for Automatic Mixed Precision
    checkpointer=dict(period=5000, max_to_keep=100),  # options for PeriodicCheckpointer
    eval_period=5000,
    log_period=20,
    dist=dict(
        num_gpus_per_node=1,
        num_nodes=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    ),
    # NCCL fusion threshold megabytes, set to 0 to compatible with previous version of OneFlow
    nccl_fusion_threshold_mb=16,
    # Maximum number of ops of NCCL fusion, set to 0 to compatible with previous version of OneFlow
    nccl_fusion_max_ops=24,
    enable_use_compute_stream=True,
)
