from omegaconf import DictConfig
from libai.config import LazyCall
from projects.GLM.modeling_glm import GLMModel


cfg = dict(
    num_layers=2,
    vocab_size=30592,
    hidden_size=768,
    num_attention_heads=16,
    max_sequence_length=512,
    embedding_dropout_prob=0.1,
    attention_dropout_prob=0.1,
    output_dropout_prob=0.1,
    layernorm_epsilon=1e-5,
    initializer_range=0.02,
    use_scaled_init_for_output_weights=True,
    bias_gelu_fusion=False,
    bias_dropout_fusion=False,
    scale_mask_softmax_fusion=False,
    apply_query_key_layer_scaling=False,
    amp_enabled=False,
    block_position_encoding=True,
    attention_scale=1.0,
    padding_idx=None,
)

cfg = DictConfig(cfg)

glm_model = LazyCall(GLMModel)(cfg=cfg)

# pretrain_model = LazyCall(GLMForPreTraining)(cfg=cfg)
