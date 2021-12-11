train = dict(
    output_dir="./output",
    init_checkpoint="",
    max_iter=5,
    amp=dict(enabled=False),  # options for Automatic Mixed Precision
    checkpointer=dict(period=5000, max_to_keep=100),  # options for PeriodicCheckpointer
    eval_period=5000,
    log_period=20,
    device="cuda"
    # ...
)
