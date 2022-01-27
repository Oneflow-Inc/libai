from libai.config import LazyCall
from omegaconf import OmegaConf
from configs.common.models.bert import pretrain_model as model
from configs.common.train import train
from configs.common.optim import optim
from configs.common.data.bert_dataset import tokenization
from libai.data import build_nlp_test_loader, build_image_train_loader
from projects.QQP.dataset.qqp_dataset import QQPDataset
from libai.tokenizer.tokenizer import _BertCNWWMTokenizer
from configs.common.models.bert import cfg as qqp_cfg
from projects.QQP.modeling.model import Classification
from projects.QQP.modeling.model import ClassificationGraph


tokenization.tokenizer = LazyCall(_BertCNWWMTokenizer)(
    vocab_file="/home/chengpeng/data/PrimeLM/data/bert-base-chinese-vocab.txt",
    lower_case=True,
)
tokenization.append_eod = False
tokenization.make_vocab_size_divisible_by = 128

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_image_train_loader)(
    dataset=[
        LazyCall(QQPDataset)(
            name="QQP_TRAIN",
            datapaths=["/home/chengpeng/train.tsv",],
            max_seq_length=512,
        ),
    ],
    num_workers=4,
)
dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(QQPDataset)(
            name="QQP_TEST",
            datapaths=["/home/chengpeng/dev.tsv",],
            max_seq_length=512,
        ),
        num_workers=4,
    ),
]

qqp_cfg.update(
    dict(
        # exist key
        hidden_size=1024,
        hidden_layers=8,
        num_attention_heads=16,
        # new key
        num_classes=2,
        pretrain_megatron_weight="/home/chengpeng/model_optim_rng.pt"
    )
)
model = LazyCall(Classification)(cfg=qqp_cfg)

train.update(
    dict(
        recompute_grad=dict(enabled=True),
        output_dir="output/finetune_qqp/",
        train_micro_batch_size=1,
        test_micro_batch_size=1,
        train_epoch=1,
        train_iter=0,
        eval_period=500,
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
    eval_graph=LazyCall(ClassificationGraph)(
        is_train=False,
        fp16=True
    )
)
# from libai.config import LazyCall as L
# from projects.QQP.modeling.model import Classification
# from configs.common.train import train
# from configs.common.optim import optim, lr_scheduler
# from configs.common.data.nlp_data import data
# from projects.QQP.modeling.model import ClassificationGraph
# from configs.common.models.bert import cfg as qqp_cfg

# # finetune model config
# # update bert cfg
# qqp_cfg.update(
#     dict(
#         # exist key
#         hidden_size=1024,
#         num_attention_heads=16,
#         # new key
#         num_classes=2,
#         pretrain_megatron_weight=None #"/home/chengpeng/model_optim_rng.pt"
#     )
# )
# model = L(Classification)(cfg=qqp_cfg)

# # add train cfg
# train.update(
#     dict(
#         output_dir="output/finetune_qqp/",
#         micro_batch_size=16,
#         global_batch_size=16,
#         train_iter=10000,
#         eval_period=10,
#         log_period=1,
#         dist=dict(
#             num_gpus_per_node=1,
#             num_nodes=1,
#             data_parallel_size=1,
#             tensor_parallel_size=1,
#             pipeline_parallel_size=1,
#             pipeline_num_layers=qqp_cfg["hidden_layers"],
#         ),
#         # new key
#         train_data=["/home/chengpeng/train.tsv",],
#         valid_data=["/home/chengpeng/demo.tsv",],
#     )
# )


# # update data cfg
# data.update(
#     dict(
#         vocab_file="/home/chengpeng/data/PrimeLM/data/bert-base-chinese-vocab.txt",
#         tokenizer_type="BertCNWWMTokenizer"
#     )
# )

# # use finetune graph
# graph = dict(
#     enabled=True,
#     train=L(ClassificationGraph)(
#         fp16=True,
#         is_eval=False,
#     ),
#     eval=L(ClassificationGraph)(
#         fp16=True,
#         is_eval=True,
#     )
# )