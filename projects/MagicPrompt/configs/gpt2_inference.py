from configs.common.models.gpt import cfg
from libai.config import LazyCall
from projects.mock_transformers import mock_tokenization
from projects.MagicPrompt.gpt2 import GPTModel, GPTForPreTraining
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
    amp_enabled=False,
    num_attention_heads=12,
    hidden_size=768,
    ffn_hidden_size=4 * 768,
    hidden_layers=12,
    max_seq_length=1024,
    initializer_range=0.02,
    vocab_size=50304,
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
    # train
    pretrained_model_path="/data/home/magicprompt",
)


model = LazyCall(GPTModel)(cfg=cfg)
pretrain_model = LazyCall(GPTForPreTraining)(cfg=cfg)
tokenization.tokenizer = LazyCall(mock_tokenization.GPT2Tokenizer)(
    vocab_file="/data/home/magicprompt/vocab.json",
    merges_file="/data/home/magicprompt/merges.txt",
)
