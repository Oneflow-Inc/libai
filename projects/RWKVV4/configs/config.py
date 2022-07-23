from omegaconf import OmegaConf

from libai.config import get_config
from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from libai.tokenizer import GPT2Tokenizer
from projects.RWKVV4.modeling.model import GPT
from projects.RWKVV4.dataset import RWKVDataset

tokenization = get_config("common/data/gpt_dataset.py").tokenization
optim = get_config("common/optim.py").optim
model_cfg = get_config("common/models/gpt.py").cfg
graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train

tokenization.tokenizer = LazyCall(GPT2Tokenizer)(
    vocab_file="/home/zhangxiaoyu/shan/RWKV-LM/RWKVV4/vocab.json",
    do_lower_case=True,
    do_chinese_wwm=False,
)
tokenization.append_eod = False
tokenization.make_vocab_size_divisible_by = 128

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(RWKVDataset)(
            data_dir="/home/zhangxiaoyu/shan/RWKV-LM/data/enwik8",
            tokenizer=tokenization.tokenizer,
            max_seq_length=128,
            mode="train",
        ),
    ],
    num_workers=4,
)
dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(RWKVDataset)(
            data_dir="/home/zhangxiaoyu/shan/RWKV-LM/data/enwik8",
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
        pretrain_megatron_weight=None,
    )
)
model = LazyCall(GPT)(cfg=model_cfg)

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
