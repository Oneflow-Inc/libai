from libai.config import LazyCall

from libai.models.t5_model import T5Model, T5ForPreTraining

cfg = dict(
    vocab_size=21128,
    hidden_size=384,
    hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=1536,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    num_tokentypes=0,
    initializer_range=0.02,
    layernorm_eps=1e-12,
    bias_gelu_fusion=True,
    bias_dropout_fusion=True,
    scale_mask_softmax_fusion=False,
    apply_query_key_layer_scaling=True,
)

bert_model = LazyCall(T5Model)(cfg=cfg)

pretrain_model = LazyCall(T5ForPreTraining)(cfg=cfg)
