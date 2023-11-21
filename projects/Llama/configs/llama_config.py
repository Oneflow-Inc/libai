from omegaconf import DictConfig, OmegaConf

from libai.config import LazyCall
from projects.Llama.llama import LlamaForCausalLM
from projects.Llama.tokenizer import LlamaTokenizer
from configs.common.train import train


cfg = dict(
    # Model
    hidden_act="silu",
    hidden_size=4096,
    initializer_range=0.02,
    intermediate_size=11008,
    max_position_embeddings=4096,
    num_attention_heads=32,
    hidden_layers=32,
    num_key_value_heads=32,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=None,
    tie_word_embeddings=False,
    dtype="float16",
    vocab_size=32000,
    use_scaled_init_for_output_weights=False,
    scale_mask_softmax_fusion=False,
    amp_enabled=False,
    # Inference
    is_encoder_decoder=False,
    max_length=20,
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
    use_cache=True,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
    # train
    pretrained_model_path=None,
)

cfg = DictConfig(cfg)

model = LazyCall(LlamaForCausalLM)(cfg=cfg)
tokenizer = LazyCall(LlamaTokenizer)(
    pretrained_model_path="/data/hf_models/meta-llama/Llama-2-7b-chat-hf/tokenizer.model"
)