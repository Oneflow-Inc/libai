import os
from omegaconf import OmegaConf

from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from libai.scheduler import WarmupExponentialLR
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader

from configs.common.train import train
from configs.common.models.graph import graph
from configs.common.optim import optim

from projects.Qwen.config.qwen_config import cfg
from projects.Qwen.utils.qwen_dataset import QwenDataset
from projects.Qwen.tokenizer import Qwen2Tokenizer
from projects.Qwen.qwen2 import Qwen2ForCausalLM


# Hyperparameters
weight_decay = 0.1
learning_rate = 5e-5
dataset_path = "/data/home/xiezipeng/libai/projects/Qwen/train_set"
pretrained_model_path = "/data/home/xiezipeng/hf_models/Qwen/Qwen1.5-7B"

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
    vocab_file="/data/home/xiezipeng/hf_models/Qwen/Qwen1.5-7B/vocab.json",
    merges_file="/data/home/xiezipeng/hf_models/Qwen/Qwen1.5-7B/merges.txt",
)


# model
model = LazyCall(Qwen2ForCausalLM)(cfg=cfg)

# datasets
dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(QwenDataset)(
            path=dataset_path, tokenizer=tokenization.tokenizer
        )
    ],
)

train.update(
    dict(
        output_dir="./sft_result",
        train_micro_batch_size=1,
        test_micro_batch_size=1,
        train_epoch=3,
        train_iter=1,
        log_period=10,
        warmup_ratio=1 / 3,
        num_accumulation_steps=8,
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
            pipeline_parallel_size=8,
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
