import os
from omegaconf import OmegaConf

from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from libai.scheduler import WarmupExponentialLR
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader

from configs.common.train import train
from configs.common.models.graph import graph
from configs.common.optim import optim

from projects.ChatGLM.configs.chatglm_config import cfg
from projects.ChatGLM.dataset import ChatGLMTrainDataset
from projects.ChatGLM.tokenizer import ChatGLMTokenizer
from projects.ChatGLM.chatglm import ChatGLMForConditionalGeneration

# Hyperparameters
weight_decay = 0.1
learning_rate = 2e-5
max_source_len = 128
max_target_len = 128
max_length = 256
dataset_path = os.environ["DATA_DIR"]
pretrained_model_path = os.environ["CHATGLM_HF_DIR"]

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
tokenization.tokenizer = LazyCall(ChatGLMTokenizer)(
    vocab_file=os.path.join(pretrained_model_path, "tokenizer.model")
)

# model
model = LazyCall(ChatGLMForConditionalGeneration)(cfg=cfg)

# datasets
dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(ChatGLMTrainDataset)(
            path=os.path.join(dataset_path, "train.json"),
            tokenizer=tokenization.tokenizer,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
            max_length=max_length,
        )
    ]
)
dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(ChatGLMTrainDataset)(
            path=os.path.join(dataset_path, "test.json"),
            tokenizer=tokenization.tokenizer,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
            max_length=max_length,
        )
    ),
]

train.update(
    dict(
        output_dir="./sft_result",
        train_micro_batch_size=1,
        test_micro_batch_size=1,
        train_epoch=3,
        train_iter=1,
        log_period=10,
        warmup_ratio=2 / 5,
        num_accumulation_steps=8,
        rdma_enabled=True,
        amp=dict(enabled=True),
        activation_checkpoint=dict(enabled=True),
        checkpointer=dict(
            period=5000,
            max_to_keep=1,
        ),
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=4,
            pipeline_num_layers=cfg.num_layers,
        ),
        evaluation=dict(
            enabled=False,
            evaluator=LazyCall(PPLEvaluator)(),
            eval_period=3000,
            eval_iter=1e5,
        ),
        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.0,
            gamma=1.0,
            warmup_method="linear",
        ),
    )
)
