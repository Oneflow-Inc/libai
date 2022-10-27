from .mt5_base import cfg
from libai.config import LazyCall
from libai.tokenizer import T5Tokenizer
from projects.MT5.mt5_model import MT5Model, MT5ForPreTraining
from configs.common.train import train
from configs.common.data.t5_dataset import tokenization

cfg.update(
    model_type="t5",
    is_encoder_decoder=True,
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
    eos_token_id=1,
    bos_token_id=None,
    sep_token_id=None,
    decoder_start_token_id=0,
)

model = LazyCall(MT5Model)(cfg=cfg)
tokenization.tokenizer = LazyCall(T5Tokenizer)(
    vocab_file="/path/to/spiece.model",
    add_bos_token=True,
)
