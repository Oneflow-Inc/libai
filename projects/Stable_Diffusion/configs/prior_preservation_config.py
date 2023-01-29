import os
import sys

from omegaconf import OmegaConf
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


class_data_root = "/home/chengpeng/chengpeng/diffusers-pytorch/examples/dreambooth/prior_dog/"
class_prompt = "a photo of dog"
train.with_prior_preservation = dict(
    enabled=True,
    class_prompt=class_prompt,
    class_data_dir=class_data_root,
    num_class_images=200,
    sample_batch_size=1,
)

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(DreamBoothDataset)(
            instance_data_root="/home/chengpeng/chengpeng/diffusers-pytorch/examples/dreambooth/demo_dog/",
            instance_prompt="a photo of sks dog",
            class_data_root=class_data_root,
            class_prompt=class_prompt,
            tokenizer=CLIPTokenizer,
            tokenizer_pretrained_folder=["CompVis/stable-diffusion-v1-4", "tokenizer"]
        )
    ],
    num_workers=4,
)


