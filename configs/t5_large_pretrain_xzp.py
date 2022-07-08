from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from libai.scheduler import WarmupExponentialLR
from projects.T5.configs.t5_config import pretrain_model as model
from .common.train import train
from projects.T5.configs.optim import optim
from .common.data.t5_dataset import dataloader, tokenization

from .common.models.graph import graph

vocab_file = "./data_test/bert_data/bert-base-chinese-vocab.txt"
data_prefix = "./data_test/bert_data/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix

# T5-large model config
model.cfg.num_attention_heads = 12
model.cfg.hidden_size = 384
model.cfg.hidden_layers = 6

train.dist.pipeline_num_layers = 2 * model.cfg.hidden_layers


train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.update(
    dict(
        output_dir="./output/t5_output",
        train_micro_batch_size=8,
        amp=dict(enabled=False),
        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.0,
            gamma=1.0,
            warmup_method="linear",
            warmup_iter=0.0,
        ),
    )
)
