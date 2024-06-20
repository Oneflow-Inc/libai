from libai.config import LazyCall
from projects.OPT.modeling.opt_model import OPTForPreTraining 

cfg = dict(
    num_layers=4,
    vocab_size=30522,
    hidden_size=128,
    ffn_hidden_size=512,
    num_attention_heads=2,
    max_seq_length=1024,
    embedding_dropout_prob=0.1,
    attention_dropout_prob=0.1,
    output_dropout_prob=0.1,
    layernorm_epsilon=1e-5,
    initializer_range=0.02,
    use_scaled_init_for_output_weights=True,
    bias_gelu_fusion=True,
    bias_dropout_fusion=True,
    scale_mask_softmax_fusion=False,
    apply_query_key_layer_scaling=True,
    apply_residual_post_layernorm=False,
    amp_enabled=False,
)

extra_cfg = dict(
    base_lr = 1.0e-3   
)

model_8m = LazyCall(OPTForPreTraining)(cfg=cfg)