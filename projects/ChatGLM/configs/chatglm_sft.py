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
max_input_length = 1350
dataset_path = "/home/lixin/DATA/CoT_zh"
pretrained_model_path = "/home/lixin/.cache/modelscope/hub/ZhipuAI/chatglm3-6b"

# graph & optim
graph["enabled"] = True
# graph['global_mode']['enabled'] = True
# graph["debug"] = 0
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
            max_len=max_input_length,
        )
    ],
    collate_fn = ChatGLMTrainDataset.collate_fn
)
dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(ChatGLMTrainDataset)(
            path=os.path.join(dataset_path, "test.json"),
            tokenizer=tokenization.tokenizer,
            max_len=max_input_length,
        ),
        collate_fn = ChatGLMTrainDataset.collate_fn
    ),
]

train.update(
    dict(
        output_dir="./sft_result",
        train_micro_batch_size=2,
        test_micro_batch_size=1,
        train_epoch=5,
        train_iter=1,
        log_period=10,
        warmup_ratio=2 / 5,
        num_accumulation_steps=8,
        rdma_enabled=True,
        amp=dict(enabled=True),
        activation_checkpoint=dict(enabled=True),
        checkpointer=dict(
            period=100,
            max_to_keep=20,
        ),
        dist=dict(
            data_parallel_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=4,
            pipeline_num_layers=cfg.num_layers,
        ),
        evaluation=dict(
            enabled=True,
            evaluator=LazyCall(PPLEvaluator)(),
            eval_period=100,
            eval_iter=1e5,
        ),
        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.0,
            gamma=1.0,
            warmup_method="linear",
        ),
    )
)
