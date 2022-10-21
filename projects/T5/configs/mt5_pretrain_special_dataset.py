from libai import evaluation
from libai.data.build import build_nlp_train_loader
from omegaconf import OmegaConf

from libai.config import LazyCall
from libai.evaluation import PPLEvaluator, evaluator
from libai.scheduler import WarmupExponentialLR

from configs.common.train import train
from configs.common.models.graph import graph

from projects.T5.configs.optim import optim
from projects.T5.configs.t5_model_config import cfg
from projects.T5.models.t5_model import T5ForPreTraining
from configs.common.data.t5_dataset import dataloader, tokenization


vocab_file = "./data_test/bert_data/bert-base-chinese-vocab.txt"
data_prefix = "./data_test/bert_data/loss_compara_content_sentence"
tokenization.tokenizer.vocab_file = vocab_file
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix

graph.debug = 1
micro_batch_size = 1
optim["lr"] = 1e-5


model = LazyCall(T5ForPreTraining)(cfg=cfg)

# model config
model.cfg.vocab_size = 12900
model.cfg.hidden_size = 1024
model.cfg.hidden_layers = 24
model.cfg.num_attention_heads = 44
model.cfg.head_size = 128
model.cfg.intermediate_size = 21504
model.cfg.hidden_dropout_prob = 0.1
model.cfg.attention_probs_dropout_prob = 0.1
model.cfg.embedding_dropout_prob = 0.1
model.cfg.layernorm_eps = 1e-6
model.cfg.model_type = "mt5"
model.cfg.pretrained_model_path = None

train.update(
    dict(
        output_dir="projects/T5/output/mt5_output",
        train_micro_batch_size=micro_batch_size,
        train_epoch=10,
        log_period=100,
        num_accumulation_steps=1,
        amp=dict(enabled=True),
        warmup_ratio=0.01,
        checkpointer=dict(period=10000, max_to_keep=10),
        dist=dict(
            data_parallel_size=2,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            pipeline_num_layers=2 * model.cfg.hidden_layers,
        ),
        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.001,
            gamma=1.0,
            warmup_method="linear",
            warmup_iter=0.0,
        ),
        evaluation=dict(
            evaluator=LazyCall(PPLEvaluator)(),
            enabled=True,
            eval_iter=20,
            eval_period=5000,
        ),
    )
)

train.zero_optimization.enabled = True
train.zero_optimization.stage = 2
