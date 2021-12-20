from ._base_.models.bert import model
from ._base_.default_train import train

# User can customize these arguments in model
model.num_embeddings = 12345
model.hidden_size = 321
model.num_layers = 3

# User can customize these arguments in train
train.eval_period = 2000
