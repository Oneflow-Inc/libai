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
from projects.SimCSE.dataset.dataset import TestDataset_sup, TrainDataset_sup
from projects.SimCSE.evaluator import SimcseEvaluator
from projects.SimCSE.modeling.simcse_sup import Simcse_sup

optim["lr"] = 1e-5
graph["enabled"] = True

tokenization.tokenizer = LazyCall(BertTokenizer)(
    vocab_file="./data/vocab.txt",
)
tokenization.make_vocab_size_divisible_by = 1


dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(TrainDataset_sup)(
            name="snli-sup",
            path="./data/SNLI/train.txt",
            tokenizer=LazyCall(BertTokenizer)(vocab_file="./data/vocab.txt"),
            max_len=64,
        )
    ],
)

dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(TestDataset_sup)(
            name="cnsd_sts",
            path="./data/STS/cnsd-sts-test.txt",
            tokenizer=LazyCall(BertTokenizer)(vocab_file="./data/vocab.txt"),
        ),
    ),
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(TestDataset_sup)(
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
        pretrained_model_weight="./data/pytorch_model.bin",
        temp=0.05,
        pooler_type="cls",
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        apply_query_key_layer_scaling=False,
        apply_residual_post_layernorm=True,
    )
)

model = LazyCall(Simcse_sup)(cfg=simcse_cfg)

train.update(
    dict(
        output_dir="./result",
        train_micro_batch_size=8,
        test_micro_batch_size=8,
        train_epoch=1,
        train_iter=1000,
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
            eval_metric="Spearman",
            eval_mode="max",
            eval_iter=100,
        ),
        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.0,
            gamma=1.0,
            warmup_method="linear",
            warmup_iter=0.0,
        ),
    )
)
