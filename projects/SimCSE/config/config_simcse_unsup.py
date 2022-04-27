from omegaconf import OmegaConf

from configs.common.data.bert_dataset import tokenization
from configs.common.models.bert import cfg as simcse_cfg
from configs.common.models.graph import graph
from configs.common.optim import optim
from configs.common.train import train
from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from libai.scheduler import WarmupExponentialLR
from libai.tokenizer import BertTokenizer
from projects.SimCSE.dataset.dataset import TestDataset_unsup, TrainDataset_unsup
from projects.SimCSE.evaluator import SimcseEvaluator
from projects.SimCSE.modeling.simcse_unsup import Simcse_unsup

optim["lr"] = 3e-5
graph["enabled"] = True

tokenization.tokenizer = LazyCall(BertTokenizer)(
    vocab_file="./data/vocab.txt",
)
tokenization.make_vocab_size_divisible_by = 1

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(TrainDataset_unsup)(
            name="snli-unsup",
            path="./data/SNLI/train.txt",
            tokenizer=LazyCall(BertTokenizer)(vocab_file="./data/vocab.txt"),
            max_len=64,
            path2="./data/STS/cnsd-sts-train.txt",
        )
    ],
)

dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(TestDataset_unsup)(
            name="cnsd_sts",
            path="./data/STS/cnsd-sts-test.txt",
            tokenizer=LazyCall(BertTokenizer)(vocab_file="./data/vocab.txt"),
        ),
    ),
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(TestDataset_unsup)(
            name="cnsd_sts",
            path="./data/STS/cnsd-sts-dev.txt",
            tokenizer=LazyCall(BertTokenizer)(vocab_file="./data/vocab.txt"),
        )
    ),
]


simcse_cfg.update(
    dict(
        vocab_size=21128,
        hidden_size=768,
        hidden_layers=12,
        layernorm_eps=1e-12,
        intermediate_size=3072,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        apply_query_key_layer_scaling=False,
        apply_residual_post_layernorm=True,
        pretrained_model_weight="./data/pytorch_model.bin",
        pooler_type="cls",
        temp=0.05,
    )
)

model = LazyCall(Simcse_unsup)(cfg=simcse_cfg)

train.update(
    dict(
        output_dir="./result",
        train_micro_batch_size=8,
        test_micro_batch_size=8,
        train_epoch=1,
        train_iter=2500,
        log_period=10,
        dist=dict(
            data_parallel_size=8,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        evaluation=dict(
            enabled=True,
            evaluator=LazyCall(SimcseEvaluator)(),
            eval_period=10,
            eval_iter=1e5,
            eval_metric="Spearman",
            eval_mode="max",
        ),
        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.000, gamma=1.0, warmup_method="linear", warmup_iter=0
        ),
    )
)
