from libai.config import LazyCall
from omegaconf import DictConfig

from projects.T5.models.t5_model import T5Model, T5ForPreTraining

cfg = dict(
    vocab_size=30522,
    hidden_size=768,
    hidden_layers=6,
    num_attention_heads=12,
    head_size=64,
    intermediate_size=1536,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    relative_attention_num_buckets=32,
    embedding_dropout_prob=0.1,
    num_tokentypes=0,
    initializer_range=0.02,
    layernorm_eps=1e-5,
    bias_gelu_fusion=False,
    bias_dropout_fusion=False,
    scale_mask_softmax_fusion=False,
    apply_query_key_layer_scaling=False,
    apply_residual_post_layernorm=False,
    amp_enabled=False,
    mlp_type="t5",
)

cfg = DictConfig(cfg)

t5_model = LazyCall(T5Model)(cfg=cfg)

pretrain_model = LazyCall(T5ForPreTraining)(cfg=cfg)
