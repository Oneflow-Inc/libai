# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import oneflow as flow
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from oneflow import nn
from oneflow.nn import functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from projects.mock_transformers import init_env  # noqa

LoRACrossAttnProcessor.forward = LoRACrossAttnProcessor.__call__


class StableDiffusion(nn.Module):
    def __init__(
        self,
        model_path,
        train_vae=False,
        train_text_encoder=False,
        train_with_lora=False,
    ):
        super().__init__()
        self.model_path = model_path
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")

        self.noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        for name in self.noise_scheduler.__dict__.keys():
            if flow.is_tensor(getattr(self.noise_scheduler, name)):
                setattr(
                    self.noise_scheduler,
                    name,
                    getattr(self.noise_scheduler, name).to_global(
                        sbp=flow.sbp.broadcast, placement=flow.env.all_device_placement("cuda")
                    ),
                )
        if not train_with_lora:
            if not train_vae:
                self.vae.requires_grad_(False)
            if not train_text_encoder:
                self.text_encoder.requires_grad_(False)
        else:
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.unet.requires_grad_(False)

            # Set correct lora layers
            lora_attn_procs = {}
            for name in self.unet.attn_processors.keys():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else self.unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]

                lora_attn_procs[name] = LoRACrossAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                )
            self.unet.set_attn_processor(lora_attn_procs)
            self.lora_layers = AttnProcsLayers(self.unet.attn_processors)

    def forward(self, pixel_values, input_ids):
        from oneflow.utils.global_view import global_mode

        placement_sbp_dict = dict(
            placement=flow.env.all_device_placement("cuda"),
            sbp=flow.sbp.split(0),
        )
        with global_mode(True, **placement_sbp_dict):
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = flow.randn(
                latents.shape, sbp=latents.sbp, placement=latents.placement, dtype=self.unet.dtype
            ).to(latents.device)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = flow.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                sbp=latents.sbp,
                placement=latents.placement,
                dtype=flow.long,
            )
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)

            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            noisy_latents = noisy_latents.to(dtype=self.unet.dtype)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(input_ids)[0]

            # Predict the noise residual
            noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

            return {"loss": loss}

    @staticmethod
    def set_activation_checkpoint(model):
        from diffusers.models.unet_2d_blocks import (
            DualTransformer2DModel,
            ResnetBlock2D,
            Transformer2DModel,
        )
        from transformers.models.clip.modeling_clip import CLIPEncoder

        for module_block in model.modules():
            prefix_name = module_block.to(nn.graph.GraphModule).name_prefix
            # unset vae checkpointing
            if prefix_name.startswith("model.vae"):
                continue
            # set clip checkpointing
            elif isinstance(module_block.to(nn.Module), CLIPEncoder):
                module_block.to(nn.graph.GraphModule).activation_checkpointing = True
            # set unet checkpointing
            elif isinstance(
                module_block.to(nn.Module),
                (ResnetBlock2D, DualTransformer2DModel, Transformer2DModel),
            ):
                module_block.to(nn.graph.GraphModule).activation_checkpointing = True
