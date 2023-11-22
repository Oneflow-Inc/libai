from omegaconf import OmegaConf

from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from libai.scheduler import WarmupExponentialLR
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader

from configs.common.train import train
from configs.common.models.graph import graph
from configs.common.optim import optim

from projects.Llama.configs.llama_config import cfg
from projects.Llama.dataset import AlpacaDataset
from projects.Llama.tokenizer import LlamaTokenizer
from projects.Llama.llama import LlamaForCausalLM


# Hyperparameters
weight_decay = 0.02
learning_rate = 3e-3
max_input_length = 1350
dataset_path = "/data/home/xiezipeng/alpaca_data/"
pretrained_model_path = "/data/hf_models/Llama-2-7b-hf"

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
    pretrained_model_path= pretrained_model_path + "/tokenizer.model"
)

# model
model = LazyCall(LlamaForCausalLM)(cfg=cfg)

# datasets
dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(AlpacaDataset)(
            path=dataset_path + "train",
            tokenizer=tokenization.tokenizer,
            max_len=max_input_length,
        )
    ],
)
dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(AlpacaDataset)(
            path=dataset_path + "test",
            tokenizer=tokenization.tokenizer,
            max_len=max_input_length,
        ),
    ),
]


train.update(
    dict(
        output_dir="./sft_result",
        train_micro_batch_size=8,
        test_micro_batch_size=8,
        train_epoch=5,
        train_iter=1,
        log_period=10,
        warmup_ratio=2/5,
        num_accumulation_steps=8,
        rdma_enabled=False,
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        evaluation=dict(
            enabled=True,
            evaluator=LazyCall(PPLEvaluator)(),
            eval_period=10,
            eval_iter=100,
        ),
        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.0,
            gamma=1.0,
            warmup_method="linear",
        ),
    )
)

