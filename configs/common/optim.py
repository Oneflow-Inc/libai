import oneflow as flow
from libai.optim import get_default_optimizer_params
from libai.scheduler import WarmupCosineAnnealingLR

from libai.config import LazyCall

optim = LazyCall(flow.optim.AdamW)(
    parameters=LazyCall(get_default_optimizer_params)(
        # parameters.model is meant to be set to the model object, before instantiating the optimizer.
        clip_grad_max_norm=1.0,
        clip_grad_norm_type=2.0,
        weight_decay_norm=0.0,
        weight_decay_bias=0.0,
    ),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    do_bias_correction=True,
)

scheduler = LazyCall(WarmupCosineAnnealingLR)(
    # max_iters is meant to be set before instantiating the scheduler, in DefaultTrainer we will automatically set this by the given train cfg. 
    warmup_factor=0, 
    # warmup_iter is meant to be set before instantiating the scheduler, in DefaultTrainer we will automatically set this by the given train cfg. 
    warmup_method="linear"
)
