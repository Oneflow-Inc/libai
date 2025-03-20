from omegaconf import DictConfig, OmegaConf

from libai.config import LazyCall
from projects.Aquila.aquila import AquilaForCausalLM
from projects.Aquila.tokenizer import AquilaTokenizer
from configs.common.train import train


cfg = dict(
    # Model
    model_type="aquila",
    hidden_act="silu",
    hidden_size=4096,
    initializer_range=0.02,
    intermediate_size=11008,
    max_position_embeddings=2048,
    num_attention_heads=32,
    hidden_layers=32,
    pretraining_tp=1,
    rms_norm_eps=1e-06,
    rope_scaling=None,
    tie_word_embeddings=False,
    vocab_size=100008,
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
    pretrained_model_path="/root/models/Aquila-7B",
)

cfg = DictConfig(cfg)

model = LazyCall(AquilaForCausalLM)(cfg=cfg)
tokenization = OmegaConf.create()
tokenization.make_vocab_size_divisible_by = 1
tokenization.tokenizer = LazyCall(AquilaTokenizer)(
    # vocab_file=cfg.pretrained_model_path+"/vocab.json",
    # merges_file=cfg.pretrained_model_path+"/merges.txt",
)
