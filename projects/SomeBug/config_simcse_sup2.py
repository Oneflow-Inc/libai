from omegaconf import OmegaConf

from configs.common.data.bert_dataset import tokenization
from configs.common.models.bert import cfg as simcse_cfg
from configs.common.models.graph import graph
from configs.common.optim import optim
from configs.common.train import train

from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from libai.tokenizer import BertTokenizer

from projects.SomeBug.dataset import TrainDataset_sup, TestDataset_sup
from projects.SomeBug.simcse_sup2 import Simcse


tokenization.tokenizer = LazyCall(BertTokenizer)(
    vocab_file="./data/vocab.txt",
)

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(TrainDataset_sup)(
            name="snli-sup",
            path="./data/SNLI/dev.txt",
            tokenizer=LazyCall(BertTokenizer)(
                vocab_file="./data/vocab.txt"
            ),
            max_len=64,
        )
    ],
)




simcse_cfg.update(
    dict(
        vocab_size=21128,
        hidden_size=768,
        hidden_layers=12,
        layernorm_eps=1e-12,
        intermediate_size=3072,
        pretrained_model_weight=None,
    )
)

model = LazyCall(Simcse)(cfg=simcse_cfg)

train.update(
    dict(
        output_dir="./result",
        train_micro_batch_size=32,
        test_micro_batch_size=32,
        train_epoch=1,
        train_iter=10000,
        eval_period=100,
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
    )
)
