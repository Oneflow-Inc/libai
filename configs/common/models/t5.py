from libai.config import LazyCall

from libai.models import T5Model, T5ForPreTraining

cfg = dict(
    vocab_size=30522,
    hidden_size=768,
    hidden_layers=6,
    num_attention_heads=16,
    intermediate_size=1536,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    embedding_dropout_prob=0.1,
    num_tokentypes=0,
    initializer_range=0.02,
    layernorm_eps=1e-5,
    bias_gelu_fusion=False,
    bias_dropout_fusion=False,
    scale_mask_softmax_fusion=False,
    apply_query_key_layer_scaling=True,
    amp_enabled=False,
)

bert_model = LazyCall(T5Model)(cfg=cfg)

pretrain_model = LazyCall(T5ForPreTraining)(cfg=cfg)
