import oneflow as flow

from libai.optim import get_default_optimizer_params
from libai.config import LazyCall

optim = LazyCall(flow.optim.AdamW)(
    params=LazyCall(get_default_optimizer_params)(
        # params.model is meant to be set to the model object,
        # before instantiating the optimizer.
        clip_grad_max_norm=1.0,
        clip_grad_norm_type=2.0,
        weight_decay_norm=0.0,
        weight_decay_bias=0.0,
    ),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
    do_bias_correction=True,
)
