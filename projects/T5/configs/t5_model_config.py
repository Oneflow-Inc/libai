from omegaconf import DictConfig


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
    initializer_range=0.02,
    layernorm_eps=1e-5,
    amp_enabled=False,
    model_type="t5",
)

cfg = DictConfig(cfg)
