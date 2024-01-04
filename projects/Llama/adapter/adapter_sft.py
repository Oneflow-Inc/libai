import os
from omegaconf import OmegaConf

from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from libai.scheduler import WarmupExponentialLR
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader

from configs.common.train import train
from configs.common.models.graph import graph
from configs.common.optim import optim

from projects.Llama.adapter.adapter_config import cfg
from projects.Llama.adapter.dataset import AlpacaDataset
from projects.Llama.tokenizer import LlamaTokenizer
from projects.Llama.adapter.adapter_model import LlamaForCausalLM


# Hyperparameters
weight_decay = 0.1
learning_rate = 2e-5
max_input_length = 512
dataset_path = "/data/home/xiezipeng/datasets/alpaca_data.json"
pretrained_model_path = "/data/home/xiezipeng/hf_models/meta-llama/Llama-2-7b-hf/"

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
tokenization.tokenizer = LazyCall(LlamaTokenizer)(
    pretrained_model_path=os.path.join(pretrained_model_path, "tokenizer.model")
)

# model
cfg.use_cache = False
model = LazyCall(LlamaForCausalLM)(cfg=cfg)

# datasets
dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(AlpacaDataset)(
            path=dataset_path,
            tokenizer=tokenization.tokenizer,
            max_len=max_input_length,
            partition="train"
        )
    ],
)
dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(AlpacaDataset)(
            path=dataset_path,
            tokenizer=tokenization.tokenizer,
            max_len=max_input_length,
            partition="test"
        ),
    ),
]


train.update(
    dict(
        output_dir="./sft_result",
        train_micro_batch_size=8,
        test_micro_batch_size=1,
        train_epoch=5,
        train_iter=1,
        log_period=10,
        warmup_ratio=2 / 5,
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
            enabled=True,
            evaluator=LazyCall(PPLEvaluator)(),
            eval_period=100,
            eval_iter=100,
        ),
        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.0,
            gamma=1.0,
            warmup_method="linear",
        ),
    )
)
