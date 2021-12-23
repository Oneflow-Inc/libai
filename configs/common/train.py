train = dict(
    output_dir="./demo_output/test_config",
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
    
    
    
nccl_fusion_threshold_mb=True,
nccl_fusion_max_ops=True,
enable_use_compute_stream=True,
)
