from omegaconf import DictConfig

from libai.config import LazyCall
from libai.scheduler import WarmupDecayLR

# fmt: off
train = dict(

    amp = dict(enabled=False),
    rdma_enabled = False,
    seed = 1234,
    model_load_dir = None,
    model_save_dir = None,

    save_initial_model = False,
    save_model_after_each_eval = False,
    learning_rate = 24,
    train_iter = 75696, #train_batches 
    #train_epoch = 0,
    log_period = 1000, #loss_print_interval
    loss_scale_policy = "static",

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

    # Save a model checkpoint after every this number of iterations,
    # and maximum number of checkpoint will be kept.
    checkpointer=dict(period=5000, max_to_keep=100, save_model_after_n_epoch=None),

    # lr_scheduler arguments
    # See flowctr/scheduler/lr_scheduler.py for definition.
    # lr_scheduler arguments
    # See flowctr/scheduler/lr_scheduler.py for definition.
    scheduler=LazyCall(WarmupDecayLR)(
        warmup_iter = 2750,
        decay_iter = 27772,
        decay_start = 49315,
    ),


    # the device type of input tensors for model, defaults to "cuda".
    # if you want to accelerate the model training when pipeline_parallel > 1
    # you can set `input_placement_device="cpu"` then call input_tensor.to_global()
    # inside your model.forward() method
    # see `flowctr/models/bert_model.py` as reference
    input_placement_device="cuda",
)
# fmt: on

train = DictConfig(train)
