from omegaconf import DictConfig
from libai.config import LazyCall
from projects.GLM.modeling_glm import GLMModel


cfg = dict(
    num_layers=48,
    vocab_size=30592,
    hidden_size=4096,
    num_attention_heads=64,
    max_sequence_length=1024,
    embedding_dropout_prob=0.0,
    attention_dropout_prob=0.0,
    output_dropout_prob=0.0,
    layernorm_epsilon=1e-5,
    initializer_range=0.02,
    use_scaled_init_for_output_weights=True,
    bias_gelu_fusion=True,
    bias_dropout_fusion=True,
    scale_mask_softmax_fusion=False,
    apply_query_key_layer_scaling=False,
    amp_enabled=True,
    block_position_encoding=True,
    attention_scale=1.0,
    padding_idx=None,
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
    use_cache=False,
    # Tokenizer
    pad_token_id=50000,
    eos_token_id=50007,
    bos_token_id=None,
    sep_token_id=None,
    decoder_start_token_id=None,
)

cfg = DictConfig(cfg)

glm_model = LazyCall(GLMModel)(cfg=cfg)
