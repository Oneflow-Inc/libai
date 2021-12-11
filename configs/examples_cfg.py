# from ._base_.models.bert import model
from ._base_.models.gpt import model
from ._base_.default_train import train

# User can customize these arguments in model
model.embedding.num_embeddings = 12345
model.blocks.hidden_size = 321
model.blocks.num_layers = 3

# User can customize these arguments in train
train.eval_period = 2000
