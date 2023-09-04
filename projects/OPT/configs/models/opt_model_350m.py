from libai.config import LazyCall
from projects.OPT.modeling.opt_model import OPTForPreTraining 
from .opt_model_8m import cfg

cfg.update(
    dict(
        num_layers=24,
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_attention_heads=16
    )
)

extra_cfg = dict(
    base_lr = 3.0e-4   
)

model_350m = LazyCall(OPTForPreTraining)(cfg=cfg)