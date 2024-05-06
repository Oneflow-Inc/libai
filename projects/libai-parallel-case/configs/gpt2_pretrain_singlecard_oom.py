from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from configs.common.models.gpt import pretrain_model as model
from configs.common.train import train
from configs.common.optim import optim
from configs.common.data.gpt_dataset import dataloader, tokenization

from configs.common.models.graph import graph

vocab_file = "projects/libai-parallel-case/gpt2/vocab.json"
merge_files = "projects/libai-parallel-case/gpt2/merges.txt"
data_prefix = "YOUR_PATH/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
tokenization.tokenizer.merges_file = merge_files
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix
dataloader.test[0].dataset.data_prefix = data_prefix
dataloader.test[0].dataset.indexed_dataset.data_prefix = data_prefix

# GPT-2 model config
model.cfg.embedding_dropout_prob = 0.1
model.cfg.attention_dropout_prob = 0.1
model.cfg.hidden_size = 2048
model.cfg.hidden_layers = 50
model.cfg.ffn_hidden_size = 2048*4
model.cfg.num_attention_heads = 32

train.input_placement_device = "cpu"

train.dist.pipeline_num_layers = model.cfg.hidden_layers

for ds in dataloader.train.dataset:
    ds.max_seq_length = model.cfg.max_seq_length

optim.lr = 1.5e-4

train.train_micro_batch_size = 1
train.amp.enabled = True

train.update(
    dict(
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            pipeline_num_layers=model.cfg.hidden_layers,
        ),
    )
)

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

graph.update(
    dict(
    # options for graph or eager mode
    enabled=True,
    debug=-1,  # debug mode for graph
    auto_parallel=dict(
        enabled=True,
        enable_auto_parallel_ignore_user_sbp_config=False,  # ignore all .to_global() in graph
        trunk_algo=True,  # consider overlapping calculate time and transfer time
        sbp_collector=False,  # use proxy node when one node transfer to many nodes
    ),
    global_mode=dict(
        enabled=False,
    ),
)
)

train.output_dir = "./output/gpt2_output"
