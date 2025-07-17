import os
import json
from omegaconf import OmegaConf

from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from libai.scheduler import WarmupExponentialLR

from configs.common.train import train
from configs.common.models.graph import graph
from configs.common.optim import optim

from projects.Llama.configs.llama_config import cfg
from projects.Llama.tokenizer import LlamaTokenizer
from projects.Llama.llama import LlamaForCausalLM

from configs.common.data.gpt_dataset import dataloader


# Hyperparameters
weight_decay = 0.1
learning_rate = 5e-5
dataset_path = "data"
pretrained_model_path = os.environ.get("PRETRAINED_MODEL_PATH", "")
cfg["pretrained_model_path"] = pretrained_model_path
print("pretrained_model_path=", pretrained_model_path)
with open(os.path.join(pretrained_model_path, "config.json"), "r", encoding="utf-8") as f:
    json_config = json.load(f)
json_cfg = OmegaConf.create(json_config)
cfg = OmegaConf.merge(cfg, json_cfg)
cfg["hidden_layers"] = json_config["num_hidden_layers"]

# graph & optim
graph["enabled"] = True
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
model = LazyCall(LlamaForCausalLM)(cfg=cfg)

# datasets
data_prefix = "data/oscar-en-10k-meg-llama_text_document"
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix
dataloader.train.dataset[0].max_seq_length = 4096
dataloader.test[0].dataset.data_prefix = data_prefix
dataloader.test[0].dataset.indexed_dataset.data_prefix = data_prefix
dataloader.test[0].dataset.max_seq_length = 4096

train.update(
    dict(
        output_dir="./sft_result",
        train_micro_batch_size=4,
        test_micro_batch_size=1,
        train_epoch=3,
        train_iter=1,
        log_period=10,
        warmup_ratio=1 / 3,
        num_accumulation_steps=8,
        rdma_enabled=False,
        amp=dict(
            enabled=True,
            dtype="bfloat16",
        ),
        activation_checkpoint=dict(enabled=True),
        checkpointer=dict(
            period=5000,
            max_to_keep=20,
        ),
        dist=dict(
            data_parallel_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
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
