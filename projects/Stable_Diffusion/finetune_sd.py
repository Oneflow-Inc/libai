import os
import oneflow as flow
from oneflow import nn
# import argparse
# from pytorch_lightning import (
#     LightningModule,
#     Trainer,
# )
# from pytorch_lightning.callbacks import (
#     LearningRateMonitor,
# )
# from fengshen.data.universal_datamodule import UniversalDataModule
# from fengshen.models.model_utils import (
#     add_module_args,
#     configure_optimizers,
#     get_total_steps,
# )
# from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from transformers import CLIPTokenizer
from transformers import OneFlowCLIPTextModel as CLIPTextModel
from diffusers import (
    OneFlowAutoencoderKL, 
    OneFlowStableDiffusionPipeline, 
    OneFlowUNet2DConditionModel,
    OneFlowDDPMScheduler
)
from oneflow.nn import functional as F
# from fengshen.data.taiyi_stable_diffusion_datasets.taiyi_datasets import add_data_args, load_data


class StableDiffusion(nn.Module):
    def __init__(self, model_path):
        super().__init__()
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

        self.text_encoder.train()
        self.vae.eval()
        self.unet.eval()

    def forward(self, pixel_values, input_ids):
        with flow.no_grad():
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
        # with torch.no_grad():
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

        return {"loss": loss}
        # self.log("train_loss", loss.item(),  on_epoch=False, prog_bar=True, logger=True)

        # if self.trainer.global_rank == 0 and self.global_step == 100:
        #     # 打印显存占用
        #     from fengshen.utils.utils import report_memory
        #     report_memory('stable diffusion')

        # if self.trainer.global_rank == 0:
        #     if (self.global_step+1) % 5000 == 0:
        #         print('saving model...')
        #         pipeline = StableDiffusionPipeline.from_pretrained(
        #             args.model_path, text_encoder=self.text_encoder, tokenizer=self.tokenizer,
        #         )
        #         self.trainer.current_epoch
        #         pipeline.save_pretrained(os.path.join(
        #             args.default_root_dir, f'hf_out_{self.trainer.current_epoch}'))

        # return {"loss": loss}

    # def setup(self, stage) -> None:
    #     if stage == 'fit':
    #         self.total_steps = get_total_steps(self.trainer, self.hparams)
    #         print('Total steps: {}' .format(self.total_steps))

    # def configure_optimizers(self):
    #     model_params = [{'params': self.text_encoder.parameters()}]
    #     if self.hparams.train_whole_model:
    #         model_params.append({'params': self.unet.parameters()})
    #     return configure_optimizers(self, model_params=model_params)

    # def training_step(self, batch, batch_idx):
    #     self.text_encoder.train()

    #     with torch.no_grad():
    #         latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
    #         latents = latents * 0.18215

    #     # Sample noise that we'll add to the latents
    #     noise = torch.randn(latents.shape).to(latents.device)
    #     noise = noise.to(dtype=self.unet.dtype)
    #     bsz = latents.shape[0]
    #     # Sample a random timestep for each image
    #     timesteps = torch.randint(
    #         0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    #     timesteps = timesteps.long()
    #     # Add noise to the latents according to the noise magnitude at each timestep
    #     # (this is the forward diffusion process)

    #     noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
    #     noisy_latents = noisy_latents.to(dtype=self.unet.dtype)

    #     # Get the text embedding for conditioning
    #     # with torch.no_grad():
    #     encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

    #     # Predict the noise residual
    #     noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

    #     loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
    #     self.log("train_loss", loss.item(),  on_epoch=False, prog_bar=True, logger=True)

    #     if self.trainer.global_rank == 0 and self.global_step == 100:
    #         # 打印显存占用
    #         from fengshen.utils.utils import report_memory
    #         report_memory('stable diffusion')

    #     if self.trainer.global_rank == 0:
    #         if (self.global_step+1) % 5000 == 0:
    #             print('saving model...')
    #             pipeline = StableDiffusionPipeline.from_pretrained(
    #                 args.model_path, text_encoder=self.text_encoder, tokenizer=self.tokenizer,
    #             )
    #             self.trainer.current_epoch
    #             pipeline.save_pretrained(os.path.join(
    #                 args.default_root_dir, f'hf_out_{self.trainer.current_epoch}'))

    #     return {"loss": loss}

    # def on_train_epoch_end(self):
    #     if self.trainer.global_rank == 0:
    #         print('saving model...')
    #         pipeline = StableDiffusionPipeline.from_pretrained(
    #             args.model_path, text_encoder=self.text_encoder, tokenizer=self.tokenizer,
    #         )
    #         self.trainer.current_epoch
    #         pipeline.save_pretrained(os.path.join(
    #             args.default_root_dir, f'hf_out_{self.trainer.current_epoch}'))

    # def on_load_checkpoint(self, checkpoint) -> None:
    #     # 兼容低版本lightning，低版本lightning从ckpt起来时steps数会被重置为0
    #     global_step_offset = checkpoint["global_step"]
    #     if 'global_samples' in checkpoint:
    #         self.consumed_samples = checkpoint['global_samples']
    #     self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset


# if __name__ == '__main__':
#     args_parser = argparse.ArgumentParser()
#     args_parser = add_module_args(args_parser)
#     args_parser = add_data_args(args_parser)
#     args_parser = UniversalDataModule.add_data_specific_args(args_parser)
#     args_parser = Trainer.add_argparse_args(args_parser)
#     args_parser = StableDiffusion.add_module_specific_args(args_parser)
#     args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
#     args = args_parser.parse_args()

#     model = StableDiffusion(args)
#     tokenizer = model.tokenizer
#     datasets = load_data(args, tokenizer=tokenizer)

#     def collate_fn(examples):
#         # print(examples)
#         input_ids = [example["instance_prompt_ids"] for example in examples]
#         pixel_values = [example["instance_images"] for example in examples]

#         pixel_values = torch.stack(pixel_values)
#         pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

#         input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True,
#                                   return_tensors="pt").input_ids
#         batch = {
#             "input_ids": input_ids,
#             "pixel_values": pixel_values,
#         }

#         return batch

#     datamoule = UniversalDataModule(
#         tokenizer=tokenizer, collate_fn=collate_fn, args=args, datasets=datasets)

#     lr_monitor = LearningRateMonitor(logging_interval='step')
#     checkpoint_callback = UniversalCheckpoint(args)

#     trainer = Trainer.from_argparse_args(args,
#                                          callbacks=[
#                                              lr_monitor,
#                                              checkpoint_callback])

#     trainer.fit(model, datamoule, ckpt_path=args.load_ckpt_path)