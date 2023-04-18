from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from projects.mock_gpt_train.configs.gpt import pretrain_model as model
from projects.MagicPrompt.configs.gpt2_dataset import dataloader, tokenization
from configs.common.optim import optim

from libai.scheduler import WarmupExponentialLR

from configs.common.train import train
from configs.common.models.graph import graph

graph.global_mode.enabled = True
# graph.enabled = False
vocab_file = "/data/home/magicprompt/vocab.json"
merge_files = "/data/home/magicprompt/merges.txt"
train_data_prefix = "/data/home/magicprompt/train/en_train_mmap_text_sentence"

tokenization.tokenizer.vocab_file = vocab_file
tokenization.tokenizer.merges_file = merge_files
dataloader.train.dataset[0].data_prefix = train_data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = train_data_prefix

train.input_placement_device = "cpu"

train.dist.pipeline_num_layers = 12

for ds in dataloader.train.dataset:
    ds.max_seq_length = 1024

optim.lr = 5.0e-05

train.update(
    dict(
        output_dir="projects/MagicPrompt/oneflow_magicprompt",
        train_micro_batch_size=4,
        test_micro_batch_size=4,
        train_epoch=33,
        train_iter=10000,
        log_period=50,
        amp=dict(enabled=False),
        warmup_ratio=0,
        checkpointer=dict(period=8000, max_to_keep=20),
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            # pipeline_num_layers=model.cfg.hidden_layers,
        ),
        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.0,
            gamma=1.0,
            warmup_method="linear",
            warmup_iter=0.0,
        ),
        evaluation=dict(
            enabled=False,
            evaluator=LazyCall(PPLEvaluator)(),
            eval_iter=250,
            eval_period=4000,
        ),
        rdma_enabled=False,
    )
)
