from omegaconf import OmegaConf

from libai.config import get_config
from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from libai.tokenizer import BertTokenizer
from projects.text_classification.modeling.model import ModelForSequenceClassification
from projects.text_classification.dataset import ClueDataset

tokenization = get_config("common/data/bert_dataset.py").tokenization
optim = get_config("common/optim.py").optim
model_cfg = get_config("common/models/bert.py").cfg
graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train

tokenization.tokenizer = LazyCall(BertTokenizer)(
    vocab_file="/DATA/disk1/liuchi/work/bert-base-chinese-vocab.txt",
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
            data_dir="./projects/text_classification/dataset/clue_data/afqmc",
            tokenizer=tokenization.tokenizer,
            max_seq_length=128,
            mode="train",
        ),
    ],
    num_workers=4,
)
dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(ClueDataset)(
            task_name="afqmc",
            data_dir="./projects/text_classification/dataset/clue_data/afqmc",
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
        pretrain_megatron_weight=None,
    )
)
model = LazyCall(ModelForSequenceClassification)(cfg=model_cfg)

train.update(
    dict(
        recompute_grad=dict(enabled=True),
        output_dir="output/benchmark/",
        train_micro_batch_size=4,
        test_micro_batch_size=4,
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
