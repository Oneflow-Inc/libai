from omegaconf import OmegaConf

from configs.common.data.bert_dataset import tokenization
from configs.common.models.bert import cfg as model_cfg
from configs.common.optim import optim
from configs.common.train import train
from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from libai.tokenizer import BertTokenizer
from projects.text_classification.modeling.model import ModelForSequenceClassification, GraphForSequenceClassification
from projects.text_classification.dataset import ClueDataset

tokenization.tokenizer = LazyCall(BertTokenizer)(
    vocab_file="bert-base-chinese-vocab.txt",
    do_lower_case=True,
    do_chinese_wwm=False,
)
tokenization.append_eod = False
tokenization.make_vocab_size_divisible_by = 128

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(ClueDataset)(
            task_name="afqmc",
            data_dir="CLUEdatasets/afqmc",
            tokenizer=tokenization.tokenizer,
            max_seq_length=512,
            mode="train",
        ),
    ],
    num_workers=4,
)
dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(ClueDataset)(
            task_name="afqmc",
            data_dir="CLUEdatasets/afqmc",
            tokenizer=tokenization.tokenizer,
            max_seq_length=512,
            mode="dev",
        ),
        num_workers=4,
    ),
]

model_cfg.update(
    dict(
        # exist key
        vocab_size=21248,
        hidden_size=1024,
        hidden_layers=24,
        num_attention_heads=16,
        # new key
        num_classes=2,
        pretrain_megatron_weight="/home/dangkai/model_optim_rng.pt",
    )
)
model = LazyCall(ModelForSequenceClassification)(cfg=model_cfg)

train.update(
    dict(
        recompute_grad=dict(enabled=True),
        output_dir="output/benchmark/",
        train_micro_batch_size=8,
        test_micro_batch_size=2,
        train_epoch=1,
        train_iter=0,
        eval_period=500,
        log_period=50,
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
    )
)

graph = dict(
    enabled=True,
    train_graph=LazyCall(GraphForSequenceClassification)(
        is_train=True,
        recompute_grad=True,
        fp16=True,
    ),
    eval_graph=LazyCall(GraphForSequenceClassification)(is_train=False, fp16=True),
)
