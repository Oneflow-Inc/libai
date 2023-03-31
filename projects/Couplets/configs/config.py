import os
import sys

dir_path = os.path.abspath(os.path.dirname(__file__))
dir_path = "/".join(dir_path.split("/")[:-1])
sys.path.append(dir_path)

from omegaconf import OmegaConf  # noqa

from dataset.dataset import CoupletsDataset  # noqa
from modeling.model import Seq2Seq  # noqa

from libai.config import get_config  # noqa
from libai.config import LazyCall  # noqa
from libai.data.build import build_nlp_train_loader, build_nlp_test_loader  # noqa

optim = get_config("common/optim.py").optim
graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(CoupletsDataset)(
            path="data_test/couplets",
            is_train=True,
            maxlen=64,
        )
    ],
    num_workers=4,
)
dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(CoupletsDataset)(
            path="data_test/couplets",
            is_train=False,
            maxlen=64,
        ),
        num_workers=4,
    )
]

transformer_cfg = dict(
    vocab_size=9027,
    max_position_embeddings=64,
    hidden_size=512,
    intermediate_size=512,
    hidden_layers=6,
    num_attention_heads=8,
    embedding_dropout_prob=0.1,
    hidden_dropout_prob=0.1,
    attention_dropout_prob=0.1,
    initializer_range=0.02,
    layernorm_epsilon=1e-5,
    bias_gelu_fusion=False,
    bias_dropout_fusion=False,
    scale_mask_softmax_fusion=False,
    apply_query_key_layer_scaling=True,
)
model = LazyCall(Seq2Seq)(cfg=transformer_cfg)

train.update(
    dict(
        rdma_enabled=False,
        activation_checkpoint=dict(enabled=False),
        amp=dict(enabled=False),
        output_dir="output/couplet/",
        train_micro_batch_size=128,
        test_micro_batch_size=32,
        train_epoch=20,
        train_iter=0,
        log_period=10,
        warmup_ratio=0.01,
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            pipeline_stage_id=None,
            pipeline_num_layers=model.cfg.hidden_layers * 2,
        ),
        evaluation=dict(
            enabled=False,
        ),
    )
)
