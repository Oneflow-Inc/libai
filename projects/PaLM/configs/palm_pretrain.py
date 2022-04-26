from libai.config import LazyCall, get_config
from .models.palm_small import model
from libai.evaluation import PPLEvaluator

graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim
data = get_config("common/data/gpt_dataset.py")

dataloader = data.dataloader
tokenization = data.tokenization


vocab_file = "./projects/PaLM/gpt_dataset/gpt2-vocab.json"
merge_files = "./projects/PaLM/gpt_dataset/gpt2-merges.txt"
data_prefix = "./projects/PaLM/gpt_dataset/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
tokenization.tokenizer.merges_file = merge_files
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix

train.train_micro_batch_size = 4
train.activation_checkpoint.enabled = True

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.output_dir = "./output/palm_output"
