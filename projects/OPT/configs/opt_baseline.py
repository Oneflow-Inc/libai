from libai.config import LazyCall, get_config
from libai.evaluation import PPLEvaluator

from models.opt_model_125m import model_125m, extra_cfg

train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim
graph = get_config("common/models/graph.py").graph
dataloader = get_config("common/data/gpt_dataset.py").dataloader
tokenization = get_config("common/data/gpt_dataset.py").tokenization


vocab_file = "./data_test/gpt_data/gpt2-vocab.json"
merge_files = "./data_test/gpt_data/gpt2-merges.txt"
data_prefix = "./data_test/gpt_data/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
tokenization.tokenizer.merges_file = merge_files
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix


optim.lr = extra_cfg["base_lr"]
optim.betas=(0.9, 0.95)
optim.weight_decay=0.1

model = model_125m

train.train_micro_batch_size = 4
train.activation_checkpoint.enabled = True

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.output_dir = "./output/opt_output"