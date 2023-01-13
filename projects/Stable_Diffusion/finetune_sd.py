import os
import oneflow as flow
from oneflow import nn
from transformers import CLIPTokenizer
from transformers import OneFlowCLIPTextModel as CLIPTextModel
from diffusers import (
    OneFlowAutoencoderKL, 
    OneFlowStableDiffusionPipeline, 
    OneFlowUNet2DConditionModel,
    OneFlowDDPMScheduler
)
from oneflow.nn import functional as F


class StableDiffusion(nn.Module):
    def __init__(
            self, 
            model_path,
            train_vae=False,
            train_text_encoder=False,
        ):
        super().__init__()
        self.model_path = model_path
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder")
        self.vae = OneFlowAutoencoderKL.from_pretrained(
            model_path, subfolder="vae")
        self.unet = OneFlowUNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet")

        self.noise_scheduler = OneFlowDDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )

        for name in self.noise_scheduler.__dict__.keys():
            if flow.is_tensor(getattr(self.noise_scheduler, name)):
                setattr(
                    self.noise_scheduler, 
                    name, 
                    getattr(self.noise_scheduler, name).to_global(
                        sbp=flow.sbp.broadcast,
                        placement=flow.env.all_device_placement("cuda")
                    ),
                )
        
        if not train_vae:
            self.vae.requires_grad_(False)
        if not train_text_encoder:
            self.text_encoder.requires_grad_(False)


    def forward(self, pixel_values, input_ids):
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = flow.randn(
            latents.shape,
            sbp = latents.sbp,
            placement = latents.placement,
            dtype = self.unet.dtype
        ).to(latents.device)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = flow.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), 
            sbp = latents.sbp,
            placement = latents.placement,
            dtype = flow.long
        )
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(dtype=self.unet.dtype)
        # noisy_latents = latents

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

        return {"loss": loss}

    @staticmethod
    def set_activation_checkpoint(model):
        from transformers.models.clip.modeling_oneflow_clip import CLIPEncoder
        from diffusers.models.unet_blocks_oneflow import (
            ResnetBlock2D, 
            DualTransformer2DModel, 
            Transformer2DModel,
            UNetMidBlock2DCrossAttn
        )
        for module_block in model.modules():
            prefix_name = module_block.to(nn.graph.GraphModule).name_prefix
            # unset vae checkpointing
            if prefix_name.startswith("model.vae"):
                continue
            # set clip checkpointing
            elif isinstance(module_block.to(nn.Module), CLIPEncoder):
                module_block.to(nn.graph.GraphModule).activation_checkpointing = True
            # set unet checkpointing
            # elif isinstance(module_block.to(nn.Module), (ResnetBlock2D, DualTransformer2DModel, Transformer2DModel)):
            elif isinstance(module_block.to(nn.Module), UNetMidBlock2DCrossAttn):
                module_block.to(nn.graph.GraphModule).activation_checkpointing = True
