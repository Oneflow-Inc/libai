from .common.models.bert import pretrain_model as model
from .common.train import train

# Bert-large model config
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 1024
