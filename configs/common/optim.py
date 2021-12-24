import oneflow as flow
from libai.optim import get_default_optimizer_params, PolynomialLR

from libai.config import LazyCall as L

optim = L(flow.optim.AdamW)(
    parameters=L(get_default_optimizer_params)(
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

lr_scheduler = L(flow.optim.lr_scheduler.WarmUpLR)(
    lrsch_or_optimizer=L(PolynomialLR)(steps=1000, end_learning_rate=1.0e-5,),
    warmup_factor=0,
    warmup_iters=100,
    warmup_method="linear",
)
