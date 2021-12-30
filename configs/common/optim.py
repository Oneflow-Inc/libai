import oneflow as flow
from libai.optim import get_default_optimizer_params

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

lr_scheduler = LazyCall(flow.optim.lr_scheduler.WarmUpLR)(
    lrsch_or_optimizer=LazyCall(flow.optim.lr_scheduler.CosineDecayLR)(
        decay_steps=1000, alpha=0.1,
    ),
    warmup_factor=0,
    warmup_iters=100,
    warmup_method="linear",
)
