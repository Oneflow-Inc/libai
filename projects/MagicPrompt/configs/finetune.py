from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from projects.MagicPrompt.configs.gpt_inference import pretrain_model as model
from projects.MagicPrompt.configs.dataset import dataloader, tokenization
from projects.MagicPrompt.configs.optim import optim
from libai.scheduler import WarmupExponentialLR

from configs.common.train import train
from configs.common.models.graph import graph

# graph.enabled = False
vocab_file = "/data/home/magicprompt/vocab.json"
merge_files = "/data/home/magicprompt/merges.txt"
train_data_prefix = "/data/home/magicprompt/train/en_train_mmap_text_sentence"
# test_data_prefix = "/data/home/magicprompt/test"


tokenization.tokenizer.vocab_file = vocab_file
tokenization.tokenizer.merges_file = merge_files
dataloader.train.dataset[0].data_prefix = train_data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = train_data_prefix

# dataloader.test.dataset[0].data_prefix = test_data_prefix
# dataloader.test.dataset[0].indexed_dataset.data_prefix = test_data_prefix

# GPT-2 model config
model.cfg.embedding_dropout_prob = 0.1
model.cfg.attention_dropout_prob = 0.1
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 384
model.cfg.ffn_hidden_size = 1536
model.cfg.hidden_layers = 6
model.cfg.max_seq_length = 1024

train.input_placement_device = "cpu"

train.dist.pipeline_num_layers = model.cfg.hidden_layers

for ds in dataloader.train.dataset:
    ds.max_seq_length = model.cfg.max_seq_length

optim.lr = 5.0e-05

train.update(
    dict(
        output_dir="projects/MagicPrompt/output",
        train_micro_batch_size=4,
        test_micro_batch_size=4,
        train_epoch=0,
        train_iter=10000,
        log_period=10,
        amp=dict(enabled=True),
        warmup_ratio=0,
        checkpointer=dict(period=100000, max_to_keep=3),
        dist=dict(
            data_parallel_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            # pipeline_num_layers=2 * model.cfg.hidden_layers,
        ),
        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.0,
            gamma=1.0,
            warmup_method="linear",
            warmup_iter=0.0,
        ),
        evaluation=dict(
            enabled=True,
            evaluator=LazyCall(PPLEvaluator)(),
            eval_iter=20,
            eval_period=50,
        ),
        rdma_enabled=False,
    )
)

train.zero_optimization.enabled = True
train.zero_optimization.stage = 2
