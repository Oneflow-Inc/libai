import os
import sys

dir_path = os.path.abspath(os.path.dirname(__file__))
dir_path = "/".join(dir_path.split("/")[:-1])
sys.path.append(dir_path)

from omegaconf import OmegaConf  # noqa

from libai.config import get_config  # noqa
from libai.config import LazyCall  # noqa
from libai.data.build import build_nlp_train_loader, build_nlp_test_loader  # noqa
from .dataset import TXTDataset
from .finetune_sd import StableDiffusion
from transformers import CLIPTokenizer

optim = get_config("common/optim.py").optim
graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(TXTDataset)(
            foloder_name="/home/chengpeng/mscoco/00000",
            tokenizer=CLIPTokenizer,
            tokenizer_pretrained_folder=["CompVis/stable-diffusion-v1-4", "tokenizer"]
        )
    ],
    num_workers=4,
)


model = LazyCall(StableDiffusion)(
    model_path="CompVis/stable-diffusion-v1-4"
)

train.update(
    dict(
        rdma_enabled=True,
        recompute_grad=dict(enabled=True),
        amp=dict(enabled=True),
        output_dir="output/stable_diffusion/",
        train_micro_batch_size=1,
        test_micro_batch_size=1,
        train_epoch=20,
        train_iter=0,
        log_period=10,
        warmup_ratio=0.01,
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            pipeline_stage_id=None,
            pipeline_num_layers=None,
        ),
        evaluation=dict(
            enabled=False,
        ),
    )
)



