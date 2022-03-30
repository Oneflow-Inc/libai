from omegaconf import OmegaConf

from configs.common.data.bert_dataset import tokenization
from configs.common.models.bert import cfg as qqp_cfg
from configs.common.optim import optim
from configs.common.train import train
from configs.common.models.graph import graph
from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from projects.QQP.dataset.qqp_dataset import QQPDataset
from projects.QQP.modeling.model import Classification
from projects.QQP.tokenizer.tokenizer import _BertCNWWMTokenizer

tokenization.tokenizer = LazyCall(_BertCNWWMTokenizer)(
    vocab_file="projects/QQP/QQP_DATA/bert-base-chinese-vocab.txt",
    lower_case=True,
)
tokenization.append_eod = False
tokenization.make_vocab_size_divisible_by = 128

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(QQPDataset)(
            dataset_name="QQP_TRAIN",
            data_paths=[
                "projects/QQP/QQP_DATA/train.tsv",
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
                "projects/QQP/QQP_DATA/dev.tsv",
            ],
            max_seq_length=512,
        ),
        num_workers=4,
    ),
]

qqp_cfg.update(
    dict(
        # exist key
        vocab_size=21248,
        hidden_size=1024,
        hidden_layers=24,
        num_attention_heads=16,
        # new key
        num_classes=2,
        pretrain_megatron_weight=None,  # "path/to/model_optim_rng.pt",
    )
)
model = LazyCall(Classification)(cfg=qqp_cfg)

optim.lr = 1e-6
optim.weight_decay = 0.1

train.update(
    dict(
        activation_checkpoint=dict(enabled=True),
        amp=dict(enabled=True),
        output_dir="output/finetune_qqp/",
        train_micro_batch_size=16,
        test_micro_batch_size=4,
        train_epoch=1,
        train_iter=0,
        eval_period=100,
        log_period=10,
        warmup_ratio=0.01,
        topk=(1,),
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
    )
)
