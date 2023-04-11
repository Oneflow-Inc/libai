from omegaconf import OmegaConf
import oneflow as flow

from libai.optim import get_default_optimizer_params
from libai.config import LazyCall
from libai.data.build import build_nlp_train_loader, build_nlp_test_loader
from projects.Stable_Diffusion.dataset import DreamBoothDataset

from transformers import CLIPTokenizer
from .dreambooth_config import (
    train,
    optim,
    graph,
    model,
)

optim.lr = 2e-6
model.train_text_encoder = True

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(DreamBoothDataset)(
            instance_data_root="path/to/demo_dog/",
            instance_prompt="a photo of sks dog",
            class_data_root="/path/to/prior_dog/",
            class_prompt="a photo of dog",
            tokenizer=CLIPTokenizer,
            tokenizer_pretrained_folder=["CompVis/stable-diffusion-v1-4", "tokenizer"],
        )
    ],
    num_workers=4,
)

train.train_iter = 2000
train.log_period = 10
