from libai.config import LazyCall

from libai.models.t5_model import T5Model, T5ForPreTraining

cfg = dict(
    vocab_size=30522,
    hidden_size=768,
    hidden_layers=24,
    num_attention_heads=12,
    intermediate_size=4096,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    add_pooling_layer=True,
    initializer_range=0.02,
    layernorm_eps=1e-5,
    bias_gelu_fusion=True,
    bias_dropout_fusion=True,
    scale_mask_softmax_fusion=False,
    apply_query_key_layer_scaling=True,
)

t5_model = LazyCall(T5Model)(cfg=cfg)

pretrain_model = LazyCall(T5ForPreTraining)(cfg=cfg)
