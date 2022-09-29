from .mt5_base import cfg


cfg.model_type = "t5"
# Parameters for sequence generation
cfg.is_encoder_decoder = True
cfg.max_length = 20
cfg.min_length = 0
cfg.do_sample = False
cfg.early_stopping = False
cfg.num_beams = 1
cfg.num_beam_groups = 1
cfg.diversity_penalty = 0.0
cfg.temperature = 1.0
cfg.top_k = 50
cfg.top_p = 1.0
cfg.typical_p = 1.0
cfg.repetition_penalty= 1.0
cfg.length_penalty = 1.0
cfg.no_repeat_ngram_size = 0
cfg.encoder_no_repeat_ngram_size = 0
cfg.num_return_sequences = 1
cfg.chunk_size_feed_forward = 0
cfg.output_scores = False
cfg.return_dict_in_generate = False
cfg.forced_bos_token_id = None
cfg.forced_eos_token_id = None
cfg.remove_invalid_values = False
cfg.exponential_decay_length_penalty = None
cfg.use_cache = True

# Tokenizer
cfg.pad_token_id = 0
cfg.eos_token_id = 1
cfg.bos_token_id = None
cfg.sep_token_id = None
cfg.decoder_start_token_id = 0

cfg.pop("cfg")
cfg["cfg"] = cfg
