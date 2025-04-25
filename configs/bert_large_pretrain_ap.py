from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from .common.models.bert import pretrain_model as model
from .common.models.graph import graph
from .common.train import train
from .common.optim import optim
from .common.data.bert_dataset import dataloader, tokenization

vocab_file = "./data_test/bert_data/bert-base-chinese-vocab.txt"
data_prefix = "./data_test/bert_data/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix
dataloader.test[0].dataset.data_prefix = data_prefix
dataloader.test[0].dataset.indexed_dataset.data_prefix = data_prefix

# Bert-large model config
# model.cfg.num_attention_heads = 16
# model.cfg.hidden_size = 768
# model.cfg.hidden_layers = 8

train.input_placement_device = "cpu"

train.dist.pipeline_num_layers = model.cfg.hidden_layers

train.train_micro_batch_size = 16

train.amp.enabled = True

for ds in dataloader.train.dataset:
    ds.max_seq_length = model.cfg.max_position_embeddings

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.output_dir = "output/bert_output"
train.update(
    dict(
        dist=dict(
            data_parallel_size=8, # starts with a default configuration: data parallelism = 8
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            pipeline_num_layers=model.cfg.hidden_layers,
        ),
    )
)
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
