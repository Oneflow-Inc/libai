from omegaconf import DictConfig, OmegaConf

from libai.config import LazyCall
from projects.Qwen2.qwen2 import Qwen2ForCausalLM
from projects.Qwen2.tokenizer import Qwen2Tokenizer
from configs.train import train


cfg = dict(
    # Model
    model_type='qwen2',
    vocab_size=151936,
    hidden_size=4096,
    intermediate_size=22016,
    hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    hidden_act="silu",
    max_position_embeddings=32768,
    initializer_range=0.02,
    rms_norm_eps=1e-06,
    rope_theta=10000.0,
    attention_dropout=0.0,
    tie_word_embeddings=False,
    use_scaled_init_for_output_weights=False,
    scale_mask_softmax_fusion=False,
    amp_enabled=True,
    # Inference
    is_encoder_decoder=False,
    max_length=256,
    min_length=0,
    do_sample=False,
    early_stopping=False,
    num_beams=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    temperature=0.7,
    top_k=20,
    top_p=0.8,
    typical_p=1.0,
    repetition_penalty=1.05,
    length_penalty=1.0,
    no_repeat_ngram_size=0,
    encoder_no_repeat_ngram_size=0,
    num_return_sequences=1,
    chunk_size_feed_forward=0,
    output_scores=False,
    use_cache=True,
    bos_token_id=151643,
    eos_token_id=151645,
    pad_token_id=151643,
    # train
    pretrained_model_path="/root/models/Qwen1.5-7B-Chat",
)

cfg = DictConfig(cfg)

model = LazyCall(Qwen2ForCausalLM)(cfg=cfg)
tokenization = OmegaConf.create()
tokenization.make_vocab_size_divisible_by = 1
tokenization.tokenizer = LazyCall(Qwen2Tokenizer)(
    vocab_file="/root/models/Qwen1.5-7B-Chat/vocab.json",
    merges_file="/root/models/Qwen1.5-7B-Chat/merges.txt",
)
