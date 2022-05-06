from libai.config import LazyCall
from projects.PaLM.palm_model import PaLM

palm_cfg = dict(
    vocab_size=50304,
    dim=768,
    depth=12,
    dim_head=64,
    num_heads=12,
    ffn_mult=4,
    initializer_range=0.02,
    layernorm_eps=1e-12,
    amp_enabled=False,
)

model = LazyCall(PaLM)(cfg=palm_cfg)
