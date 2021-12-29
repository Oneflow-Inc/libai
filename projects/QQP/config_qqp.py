from libai.config import LazyCall as L
from projects.QQP.modeling.model import Classification
from configs.common.train import train
from configs.common.optim import optim, lr_scheduler
from configs.common.data.nlp_data import data
from projects.QQP.modeling.model import ClassificationGraph
from configs.common.models.bert import cfg as qqp_cfg

# finetune model config
# update bert cfg
qqp_cfg.update(
    dict(
        # exist key
        hidden_size=1024,
        num_attention_heads=16,
        # new key
        num_classes=2,
        pretrain_megatron_weight="/home/chengpeng/model_optim_rng.pt"
    )
)
model = L(Classification)(cfg=qqp_cfg)

# add train cfg
train.update(
    dict(
        output_dir="output/finetune_qqp/",
        micro_batch_size=16,
        global_batch_size=16,
        train_iter=10000,
        dist=dict(
            num_gpus_per_node=1,
            num_nodes=1,
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            pipeline_num_layers=qqp_cfg["hidden_layers"],
        ),
        # new key
        train_data=["/home/chengpeng/train.tsv",],
        valid_data=["/home/chengpeng/dev.tsv",],
    )
)


# update data cfg
data.update(
    dict(
        vocab_file="/home/chengpeng/PrimeLM/data/bert-base-chinese-vocab.txt",
        tokenizer_type="BertCNWWMTokenizer"
    )
)

# use finetune graph
graph = dict(
    enabled=True,
    train=L(ClassificationGraph)(
        fp16=True,
        is_eval=False,
    ),
    eval=L(ClassificationGraph)(
        fp16=True,
        is_eval=True,
    )
)