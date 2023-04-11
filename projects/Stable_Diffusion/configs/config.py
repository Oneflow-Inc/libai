from omegaconf import OmegaConf

from libai.config import get_config
from libai.config import LazyCall
from libai.data.build import build_nlp_train_loader, build_nlp_test_loader
from projects.Stable_Diffusion.dataset import TXTDataset
from projects.Stable_Diffusion.modeling import StableDiffusion
from transformers import CLIPTokenizer

optim = get_config("common/optim.py").optim
graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train

graph.global_mode.enabled = True

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(TXTDataset)(
            foloder_name="/path/to/mscoco/00000",
            tokenizer=CLIPTokenizer,
            tokenizer_pretrained_folder=["CompVis/stable-diffusion-v1-4", "tokenizer"],
        )
    ],
    num_workers=4,
)


model = LazyCall(StableDiffusion)(model_path="CompVis/stable-diffusion-v1-4")

train.update(
    dict(
        rdma_enabled=True,
        activation_checkpoint=dict(enabled=True),
        zero_optimization=dict(
            enabled=True,
            stage=2,
        ),
        checkpointer=dict(period=5000000),
        amp=dict(enabled=True),
        output_dir="output/stable_diffusion/",
        train_micro_batch_size=1,
        test_micro_batch_size=1,
        train_epoch=0,
        train_iter=20,
        log_period=1,
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
