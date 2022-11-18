from configs.common.models.gpt import cfg
from libai.config import LazyCall
from libai.tokenizer.tokenization_gpt2 import GPT2Tokenizer
from projects.MagicPrompt.gpt2 import GPTModel
from configs.common.data.gpt_dataset import tokenization
from configs.common.train import train


cfg.update(
    # Model
    embedding_dropout_prob=0,
    attention_dropout_prob=0,
    output_dropout_prob=0,
    bias_gelu_fusion=False,
    bias_dropout_fusion=False,
    scale_mask_softmax_fusion=False,
    apply_query_key_layer_scaling=False,
    apply_residual_post_layernorm=False,
    amp_enabled=True,
    # Inference
    is_encoder_decoder=False,
    max_length=20,
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
    pad_token_id=0,
    eos_token_id=50256,
    bos_token_id=50256,
    sep_token_id=None,
    decoder_start_token_id=None,
)


model = LazyCall(GPTModel)(cfg=cfg)
tokenization.tokenizer = LazyCall(GPT2Tokenizer)(
    vocab_file="/home/xiezipeng/libai/xzp/gpt2-sd/vocab.json",
    merges_file="/home/xiezipeng/libai/xzp/gpt2-sd/merges.txt",
    add_bos_token=True,
)
