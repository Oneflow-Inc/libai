from omegaconf import DictConfig
from libai.config import LazyCall
from libai.models import RobertaModel, RobertaForPreTraining, RobertaForCausalLM


cfg = dict(
    vocab_size=50265,
    hidden_size=768,
    hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=514,
    num_tokentypes=1,
    add_pooling_layer=True,
    initializer_range=0.02,
    layernorm_eps=1e-5,
    pad_token_id=1,
    bias_gelu_fusion=True,
    bias_dropout_fusion=True,
    scale_mask_softmax_fusion=True,
    apply_query_key_layer_scaling=True,
    apply_residual_post_layernorm=False,
    amp_enabled=False,
)

cfg = DictConfig(cfg)

roberta_model = LazyCall(RobertaModel)(cfg=cfg)

roberta_causal_lm = LazyCall(RobertaForCausalLM)(cfg=cfg)

pretrain_model = LazyCall(RobertaForPreTraining)(cfg=cfg)
