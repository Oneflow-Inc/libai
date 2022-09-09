from timeit import repeat
from typing import Dict
import numpy as np
from omegaconf import OmegaConf
import omegaconf
import oneflow as flow

from libai.config import LazyCall, instantiate
from libai.models.build import build_model
from libai.data.structures import DistTensorData, Instance
from libai.inference.basic import BasePipeline
import libai.utils.distributed as dist
from dalle2.tokenizer import SimpleTokenizer
from oneflow.framework import balanced_splitter


class Dalle2Pipeline(BasePipeline):
    def __init__(
        self,
        config_file,
        data_parallel=None,
        tensor_parallel=None,
        pipeline_parallel=None,
        pipeline_stage_id=None,
        pipeline_num_layers=None,
        model_path=None,
        mode="libai",
        **kwargs,
    ):
        super().__init__(
            config_file,
            data_parallel,
            tensor_parallel,
            pipeline_parallel,
            pipeline_stage_id,
            model_path,
            pipeline_num_layers,
            mode,
            **kwargs,
        )

    def update_cfg(
        self,
        data_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=1,
        pipeline_stage_id=None,
        pipeline_num_layers=None,
    ):
        super().update_cfg(
            data_parallel,
            tensor_parallel,
            pipeline_parallel,
            pipeline_stage_id,
            pipeline_num_layers,
        )
        self.cfg.dataloader = None
        self.cfg.tokenization = OmegaConf.create()
        self.cfg.tokenization.tokenizer = LazyCall(SimpleTokenizer)()

    def load_prior_weight(self, prior, prior_weight_path):
        if isinstance(prior, omegaconf.dictconfig.DictConfig):
            prior = build_model(prior)
        import torch
        state_dict = torch.load(prior_weight_path, map_location="cpu")['ema_model']
        for k, torch_tensor in state_dict.items():
            if "clip." in k:
                continue
            if k.endswith(".g"):
                k = k[:-1] + "weight"
            elif k.startswith("net.causal_transformer"):
                if k.endswith("gamma"):
                    k = k[:-5] + 'weight'
                elif k.endswith('beta'):
                    k = k[:-4] + 'bias'
            assert k in prior.state_dict(), k
            flow_tensor = flow.tensor(torch_tensor.cpu().numpy(), placement=prior.state_dict()[
                                      k].placement, sbp=prior.state_dict()[k].sbp)
            prior.state_dict()[k].data.copy_(flow_tensor.data)

        return prior.eval()

    def load_decoder_weight(self, decoder, decoder_weight_path):
        if isinstance(decoder, omegaconf.dictconfig.DictConfig):
            decoder = build_model(decoder)
        import torch
        state_dict = torch.load(decoder_weight_path, map_location="cpu")
        for k, torch_tensor in state_dict.items():
            if 'clip.' in k:
                continue
            if k.endswith(".g"):
                k = k[:-1] + "weight"
            elif 'cross_attn' in k:
                if k.endswith('gamma'):
                    k = k[:-5] + "weight"
                elif k.endswith('beta'):
                    k = k[:-4] + "bias"
            assert k in decoder.state_dict().keys(), k
            flow_tensor = flow.tensor(torch_tensor.cpu().numpy(), placement=decoder.state_dict()[
                                      k].placement, sbp=decoder.state_dict()[k].sbp)
            decoder.state_dict()[k].data.copy_(flow_tensor.data)
        return decoder.eval()

    def load_pretrain_weight(
        self,
        libai_cfg_model,
        model_path,
        mode='libai'
    ):
        model = build_model(libai_cfg_model)
        self.load_prior_weight(model.prior, libai_cfg_model.prior_weight_path)
        self.load_decoder_weight(model.decoder, libai_cfg_model.decoder_weight_path)
        return model

    def build_tokenizer(self, cfg):
        return SimpleTokenizer()  #return instantiate(cfg.tokenizer)

    def _parse_parameters(self, use_cache=None, max_generate_length=10, **pipeline_parameters):
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {**pipeline_parameters}

        if use_cache is not None:
            assert isinstance(use_cache, bool), "use_cache must be True or False"
            forward_params["use_cache"] = use_cache
        if max_generate_length is not None:
            assert isinstance(max_generate_length, int), "max_generate_length must be integer"
            forward_params["max_generate_length"] = max_generate_length
        return preprocess_params, forward_params, postprocess_params

    def postprocess(self, **kwargs: Dict) -> dict:
        return kwargs

    def preprocess(self, input_, **preprocess_parameters: Dict) -> dict:
        return {}

    def __call__(self, text, parallel='data') -> dict:
        if parallel == 'data':
            text = self.split_data(text)
        else:
            assert parallel == 'col'
        return self.forward(text, parallel)

    def split_data(self, text):
        rank = dist.get_rank()
        indices = balanced_splitter.BalancedRanges(len(text), dist.get_world_size())
        return text[indices[rank][0]:indices[rank][1]]

    def forward(self, text, parallel) -> dict:
        sbp = flow.sbp.split(0) if parallel == 'data' else flow.sbp.broadcast
        tokens = self.tokenizer.tokenize(text).to_global(placement=flow.placement(type='cuda', ranks=list(range(dist.get_world_size()))), sbp=sbp)
        prior = self.model.prior
        decoder = self.model.decoder
        text_embed, text_encodings, text_mask = prior.clip.embed_text(tokens)
        image_embed = prior.sample(tokens, num_samples_per_batch=2, cond_scale=1.)

        image_embed = decoder.sample(
            image_embed=image_embed, text_encodings=text_encodings, text_mask=text_mask, cond_scale=3.5)
        return {"image_embed": image_embed}

    def save_images(self, images, args):
        import flowvision.transforms as T
        to_pil = T.ToPILImage()
        images = images.to_local().to("cpu")
        images_64x64 = list(map(to_pil, [images[i] for i in range(images.shape[0])]))
        for i, image in enumerate(images_64x64):                
            image.save(f"{args.output_dir}/{i}.png")

        if args.upsample_scale:
            from swinir import load_model, upsample4x, upsample16x
            swinir = load_model(args.swinir_path)
            upsample_fun = upsample4x if args.upsample_scale == 4 else upsample16x
            images = upsample_fun(images, swinir).to('cpu')
            images = list(map(to_pil, [images[i] for i in range(images.shape[0])]))
            for i, image in enumerate(images):                
                image.save(f"{args.output_dir}/{i}_{args.upsample_scale}x.png")


def parser_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='configs/dalle2_config.py')
    parser.add_argument('--data_parallel', type=int, default=1)
    parser.add_argument('--tensor_parallel', type=int, default=4)
    parser.add_argument('--pipeline_parallel', type=int, default=1)
    parser.add_argument('--text', type=str, default="A man is playing basketball with his friends!")
    parser.add_argument('--upsample_scale', type=int, choices=[4, 16], default=None, help="upsample scale, if 4x, output resolution will be 256 x 256.")
    parser.add_argument('--swinir_path', type=str, default='./swinir/weights/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    return parser.parse_args()

if __name__ == "__main__":
    args = parser_args()
    model = Dalle2Pipeline(
        config_file=args.config_file,
        data_parallel=args.data_parallel,
        tensor_parallel=args.tensor_parallel,
        pipeline_parallel=args.pipeline_parallel)
    text = args.text if args.text else "a man is playing basketball with his friends!"
    repeats = 4
    parallel = 'data' if args.data_parallel > 1 else 'col'
    imgs = model([text] * repeats, parallel)
    imgs = imgs['image_embed']
    rank = dist.get_rank()
    if rank == 0:
        imgs = imgs.to_global(placement=flow.placement(
            type='cuda', ranks=[0]), sbp=flow.sbp.broadcast)
        model.save_images(imgs, args)

