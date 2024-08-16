from omegaconf import DictConfig
from libai.config import LazyCall
from projects.MT5.mt5_model import MT5Model, MT5ForPreTraining


cfg = dict(
    vocab_size=250112,
    hidden_size=1024,
    hidden_layers=24,
    num_attention_heads=16,
    head_size=64,
    intermediate_size=2816,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    embedding_dropout_prob=0.1,
    relative_attention_num_buckets=32,
    initializer_range=1.0,
    layernorm_eps=1e-06,
    amp_enabled=False,
    model_type="mt5",
    eos_token_id=1,
    padding_idx=0,
    is_encoder_decoder=True,
    tie_word_embeddings=False,
)

cfg = DictConfig(cfg)

mt5_model = LazyCall(MT5Model)(cfg=cfg)
pretrain_model = LazyCall(MT5ForPreTraining)(cfg=cfg)
