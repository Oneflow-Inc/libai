from omegaconf import OmegaConf

from configs.common.data.bert_dataset import tokenization
from configs.common.models.bert import cfg as simcse_cfg
from configs.common.models.graph import graph
from configs.common.optim import optim
from configs.common.train import train

from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from libai.tokenizer import BertTokenizer

from projects.SimCSE.dataset.dataset import TrainDataset, TestDataset
from projects.SimCSE.modeling.simcse_unsup import SimcseModel


tokenization.tokenizer = LazyCall(BertTokenizer)(
    vocab_file="/home/xiezipeng/libai/projects/SimCSE/data/vocab.txt"
)

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(TrainDataset)(
            name="wiki",
            path="/home/xiezipeng/libai/projects/SimCSE/data/wiki1m_for_simcse.txt",
            tokenizer=LazyCall(BertTokenizer)(
                vocab_file="/home/xiezipeng/libai/projects/SimCSE/data/vocab.txt"
            ),
            max_len=simcse_cfg["max_position_embeddings"],
        )
    ],
)

dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(TestDataset)(
            name="sts",
            path="/home/xiezipeng/libai/projects/SimCSE/data/sts_test.txt",
            tokenizer=LazyCall(BertTokenizer)(
                vocab_file="/home/xiezipeng/libai/projects/SimCSE/data/vocab.txt"
            ),
        ),
    ),
    # LazyCall(build_nlp_test_loader)(
    #     dataset=LazyCall(TestDataset)(
    #         name="sts",
    #         path="/home/xiezipeng/libai/projects/SimCSE/data/sts_dev.txt",
    #         tokenizer=LazyCall(BertTokenizer)(
    #             vocab_file="/home/xiezipeng/libai/projects/SimCSE/data/vocab.txt"
    #         ),
    #     )
    # ),
]


simcse_cfg.update(
    dict(
        # intermediate_size=3072,
        hidden_layers=12,
        # layernorm_eps=1e-12,
        # pretrained_model_weight="/home/xiezipeng/libai/projects/SimCSE/data/model_optim_rng.pt",
        pretrained_model_weight=None,
        pooler_type="cls",
        temp=1,
        hidden_size=768,
    )
)

model = LazyCall(SimcseModel)(cfg=simcse_cfg)

train.update(
    dict(
        output_dir="/home/xiezipeng/libai/projects/SimCSE/result",
        train_micro_batch_size=10,
        test_micro_batch_size=10,
        train_epoch=1,
        train_iter=10000,
        eval_period=30,
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
    )
)
