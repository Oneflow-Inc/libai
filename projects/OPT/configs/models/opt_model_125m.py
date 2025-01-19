from libai.config import LazyCall
from projects.OPT.modeling.opt_model import OPTForPreTraining 
from .opt_model_8m import cfg

cfg.update(
    dict(
        num_layers=12,
        hidden_size=768,
        ffn_hidden_size=3072,
        num_attention_heads=12
    )
)

extra_cfg = dict(
    base_lr = 6.0e-4   
)

model_125m = LazyCall(OPTForPreTraining)(cfg=cfg)