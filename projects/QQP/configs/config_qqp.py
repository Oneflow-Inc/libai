from omegaconf import OmegaConf

from configs.common.data.bert_dataset import tokenization
from configs.common.models.bert import cfg as qqp_cfg
from configs.common.optim import optim
from configs.common.train import train
from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from projects.QQP.dataset.qqp_dataset import QQPDataset
from projects.QQP.modeling.model import Classification, ClassificationGraph
from projects.QQP.tokenizer.tokenizer import _BertCNWWMTokenizer

tokenization.tokenizer = LazyCall(_BertCNWWMTokenizer)(
    vocab_file="/home/chengpeng/data/PrimeLM/data/bert-base-chinese-vocab.txt",
    lower_case=True,
)
tokenization.append_eod = False
tokenization.make_vocab_size_divisible_by = 128

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(QQPDataset)(
            name="QQP_TRAIN",
            datapaths=[
                "/home/chengpeng/train.tsv",
            ],
            max_seq_length=512,
        ),
    ],
    num_workers=4,
)
dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(QQPDataset)(
            name="QQP_TEST",
            datapaths=[
                "/home/chengpeng/dev.tsv",
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
        pretrain_megatron_weight="/home/chengpeng/model_optim_rng.pt",
    )
)
model = LazyCall(Classification)(cfg=qqp_cfg)

train.update(
    dict(
        recompute_grad=dict(enabled=True),
        output_dir="output/finetune_qqp/",
        train_micro_batch_size=16,
        test_micro_batch_size=2,
        train_epoch=1,
        train_iter=0,
        eval_period=100,
        log_period=10,
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
    )
)

graph = dict(
    enabled=True,
    train_graph=LazyCall(ClassificationGraph)(
        is_train=True,
        recompute_grad=True,
        fp16=True,
    ),
    eval_graph=LazyCall(ClassificationGraph)(is_train=False, fp16=True),
)
