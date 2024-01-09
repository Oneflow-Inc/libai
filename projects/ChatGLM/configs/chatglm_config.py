import os
from omegaconf import DictConfig, OmegaConf

from libai.config import LazyCall
from projects.ChatGLM.chatglm import ChatGLMForConditionalGeneration
from projects.ChatGLM.tokenizer import ChatGLMTokenizer
from configs.common.train import train


cfg = dict(
    # Model
    add_bias_linear=False,
    add_qkv_bias=True,
    apply_query_key_layer_scaling=True,
    apply_residual_connection_post_layernorm=False,
    attention_dropout=0.0,
    attention_softmax_in_fp32=True,
    ffn_hidden_size=13696,
    fp32_residual_connection=False,
    hidden_dropout=0.0,
    hidden_size=4096,
    kv_channels=128,
    layernorm_epsilon=1e-05,
    multi_query_attention=True,
    multi_query_group_num=2,
    num_attention_heads=32,
    num_layers=28,
    padded_vocab_size=65024,
    post_layer_norm=True,
    rmsnorm=True,
    seq_length=8192,
    use_cache=True,
    tie_word_embeddings=False,
    eos_token_id=2,
    bos_token_id=1,
    pad_token_id=0,
    pre_seq_len=None,
    prefix_projection=None,
    use_return_dict=True,
    amp_enabled=True,
    # Inference
    is_encoder_decoder=False,
    max_length=1350,
    min_length=0,
    do_sample=False,
    early_stopping=False,
    num_beams=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    temperature=0.9,
    top_k=50,
    top_p=0.6,
    typical_p=1.0,
    repetition_penalty=1.0,
    length_penalty=1.0,
    no_repeat_ngram_size=0,
    encoder_no_repeat_ngram_size=0,
    num_return_sequences=1,
    chunk_size_feed_forward=0,
    output_scores=False,
    output_hidden_states=False,
    # train
    pretrained_model_path=os.environ["CHATGLM_HF_DIR"],
    # lora_cfg
    lora_enable=False,
    lora_cfg=dict(
        # Model
        r=8,
        target_modules=["query_key_value"],
        lora_alpha=8,
        lora_dropout=0.0,
        fan_in_fan_out=False,
        bias="lora_only",
        modules_to_save=None,
        init_lora_weights=True,  # or lora
        inference_mode=False,
        rank_pattern=dict(),
        alpha_pattern=dict(),
    ),
    lora_pretrained_model_path=None,  # None for train
)

cfg = DictConfig(cfg)

model = LazyCall(ChatGLMForConditionalGeneration)(cfg=cfg)
tokenization = OmegaConf.create()
tokenization.make_vocab_size_divisible_by = 1
tokenization.tokenizer = LazyCall(ChatGLMTokenizer)(
    vocab_file=f"{os.environ['CHATGLM_HF_DIR']}/tokenizer.model"
)
