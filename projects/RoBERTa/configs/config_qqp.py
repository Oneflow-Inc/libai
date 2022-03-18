from omegaconf import OmegaConf

from configs.common.data.bert_dataset import tokenization
from configs.common.optim import optim
from configs.common.train import train
from configs.common.models.graph import graph
from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from projects.RoBERTa.configs.config_roberta import cfg as qqp_cfg
from projects.RoBERTa.dataset.qqp_dataset import QQPDataset
from projects.RoBERTa.modeling.model import Classification
from projects.RoBERTa.tokenizer.tokenizer import _BertCNWWMTokenizer

tokenization.tokenizer = LazyCall(_BertCNWWMTokenizer)(
    vocab_file="/home/zhengwen/libai/data_test/bert_data/bert-base-chinese-vocab.txt",
    lower_case=True,
)
tokenization.setup = True
tokenization.append_eod = False
tokenization.make_vocab_size_divisible_by = 128

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(QQPDataset)(
            dataset_name="QQP_TRAIN",
            data_paths=[
                "/home/zhengwen/train.tsv",
            ],
            max_seq_length=512,
        ),
    ],
    num_workers=4,
)
dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(QQPDataset)(
            dataset_name="QQP_TEST",
            data_paths=[
                "/home/zhengwen/dev.tsv",
            ],
            max_seq_length=512,
        ),
        num_workers=4,
    ),
]

# 修改roberta更改的参数！！！！
qqp_cfg.update(
    dict(
        # exist key
        # vocab_size=21248,
        # hidden_size=1024,
        # hidden_layers=24,
        # num_attention_heads=16,
        # new key
        num_classes=2,
        pretrain_megatron_weight=None,
    )
)

model = LazyCall(Classification)(cfg=qqp_cfg)

optim.lr = 1e-4
optim.weight_decay = 0.1

# 这里是自己加的~
graph.update(
    dict(
        enabled=False,
    )
)

train.update(
    dict(
        recompute_grad=dict(enabled=True),
        amp=dict(enabled=True),
        output_dir="output/finetune_qqp/",
        train_micro_batch_size=16,
        test_micro_batch_size=4,
        train_epoch=1,
        train_iter=0,
        eval_period=100,
        log_period=10,
        warmup_ratio=0.01,
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
    )
)
