from omegaconf import DictConfig
from libai.config import LazyCall
from projects.BLOOM.modeling.bloom_model import BloomModel


cfg = dict(
    # model
    vocab_size=250880,
    hidden_size=64,
    hidden_layers=2,
    n_head=8,
    padding_idx=3,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    apply_residual_connection_post_layernorm=False,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    pretraining_tp=1,
    slow_but_exact=False,
    amp_enabled=False,
    # Inference
    is_encoder_decoder=False,
    max_length=512,
    min_length=0,
    do_sample=False,
    early_stopping=False,
    num_beams=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    typical_p=1.0,
    repetition_penalty=1.0,
    length_penalty=1.0,
    no_repeat_ngram_size=0,
    encoder_no_repeat_ngram_size=0,
    num_return_sequences=1,
    chunk_size_feed_forward=0,
    output_scores=False,
    forced_bos_token_id=None,
    forced_eos_token_id=None,
    remove_invalid_values=False,
    exponential_decay_length_penalty=None,
    use_cache=True,
    # Tokenizer
    pad_token_id=3,
    eos_token_id=2,
    bos_token_id=1,
    sep_token_id=None,
    decoder_start_token_id=None,
)

cfg = DictConfig(cfg)

glm_model = LazyCall(BloomModel)(cfg=cfg)
