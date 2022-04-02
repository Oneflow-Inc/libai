import oneflow as flow

from libai.optim import get_default_optimizer_params
from libai.config import LazyCall

optim = LazyCall(flow.optim.SGD)(
    params=LazyCall(get_default_optimizer_params)(
        clip_grad_max_norm=None,
        clip_grad_norm_type=None,
        weight_decay_norm=None,
        weight_decay_bias=None,
    ),
    lr=1e-4,
    weight_decay=0.01,
    momentum = 0.9, 
)