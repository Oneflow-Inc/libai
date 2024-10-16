import os

from omegaconf import OmegaConf

from configs.common.models.graph import graph
from configs.common.optim import optim
from configs.common.train import train
from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from libai.evaluation import PPLEvaluator
from libai.scheduler import WarmupExponentialLR
from projects.Qwen.configs.qwen_config import cfg
from projects.Qwen.qwen2 import Qwen2ForCausalLM
from projects.Qwen.tokenizer import Qwen2Tokenizer
from projects.Qwen.qwen_dataset import QwenDataset

# Hyperparameters
weight_decay = 0.1
learning_rate = 5e-5
dataset_path = os.environ["DATA_DIR"]
pretrained_model_path = os.environ["MODEL_DIR"]

# graph & optim
graph["enabled"] = False
optim.update(
    dict(
        lr=learning_rate,
        weight_decay=weight_decay,
    )
)

# tokenize
tokenization = OmegaConf.create()
tokenization.make_vocab_size_divisible_by = 1
tokenization.tokenizer = LazyCall(Qwen2Tokenizer)(
    vocab_file=pretrained_model_path + "/vocab.json",
    merges_file=pretrained_model_path + "/merges.txt",
)


# model
cfg.pretrained_model_path = pretrained_model_path
model = LazyCall(Qwen2ForCausalLM)(cfg=cfg)

# datasets
dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(QwenDataset)(
            path=os.path.join(dataset_path, "train"), tokenizer=tokenization.tokenizer
        )
    ],
)
dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(QwenDataset)(
            path=os.path.join(dataset_path, "test"), tokenizer=tokenization.tokenizer
        ),
    ),
]

train.update(
    dict(
        output_dir="./sft_result/qwen",
        train_micro_batch_size=1,
        test_micro_batch_size=1,
        train_epoch=1,
        train_iter=1,
        log_period=1,
        warmup_ratio=1 / 3,
        num_accumulation_steps=1,
        rdma_enabled=False,
        amp=dict(enabled=True),
        activation_checkpoint=dict(enabled=True),
        checkpointer=dict(
            period=5000,
            max_to_keep=20,
        ),
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            pipeline_num_layers=cfg.hidden_layers,
        ),
        evaluation=dict(
            enabled=False,
            evaluator=LazyCall(PPLEvaluator)(),
            eval_period=1000,
            eval_iter=1e5,
        ),
        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.0,
            gamma=1.0,
            warmup_method="linear",
        ),
    )
)
