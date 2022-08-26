from typing import Dict
import numpy as np
from omegaconf import OmegaConf
import omegaconf
import oneflow as flow

from libai.config import LazyCall
from libai.models.build import build_model
from libai.data.structures import DistTensorData, Instance
from libai.inference.basic import BasePipeline
from dalle2.tokenizer import SimpleTokenizer


class Dalle2Pipeline(BasePipeline):
    def __init__(
        self,
        config_file,
        data_parallel=None,
        tensor_parallel=None,
        pipeline_parallel=None,
        pipeline_stage_id=None,
        model_path=None,
        **kwargs,
    ):
        super().__init__(
            config_file, 
            data_parallel, 
            tensor_parallel, 
            pipeline_parallel, 
            pipeline_stage_id, 
            model_path, 
            **kwargs
        )

    def update_cfg(
        self,
        data_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=1,
        pipeline_stage_id=None,
    ):
        super().update_cfg(
            data_parallel, 
            tensor_parallel, 
            pipeline_parallel,
            pipeline_stage_id
        )
        self.cfg.dataloader = None
        self.cfg.tokenization = OmegaConf.create()
        self.cfg.tokenization.tokenizer = LazyCall(SimpleTokenizer)()

    def load_prior_weight(self, prior, prior_weight_path):
        if isinstance(prior, omegaconf.dictconfig.DictConfig):
            prior = build_model(prior)#.to_global(placement=flow.placement(type='cuda',ranks=[0, 1, 2, 3]), sbp=flow.sbp.broadcast)
        import torch
        state_dict = torch.load(prior_weight_path, map_location="cpu")['ema_model']
        for k,torch_tensor in state_dict.items():
            if "clip." in k:
                continue
            if k.endswith(".g"):
                k= k[:-1] + "weight" 
            elif k.startswith("net.causal_transformer"):
                if k.endswith("gamma"):
                    k= k[:-5] + 'weight'
                elif k.endswith('beta'):
                    k = k[:-4] + 'bias'
            assert k in prior.state_dict(), k
            flow_tensor = flow.tensor(torch_tensor.cpu().numpy(), placement=prior.state_dict()[k].placement, sbp=prior.state_dict()[k].sbp)
            prior.state_dict()[k].data.copy_(flow_tensor.data)
            
        return prior.eval()

    def load_decoder_weight(self, decoder, decoder_weight_path):
        if isinstance(decoder, omegaconf.dictconfig.DictConfig):
            decoder = build_model(decoder)#.to_global(placement=flow.placement(type='cuda',ranks=[0, 1, 2, 3]), sbp=flow.sbp.broadcast).eval()
        import torch
        state_dict = torch.load(decoder_weight_path, map_location = "cpu")
        for k, torch_tensor in state_dict.items():
            if 'clip.' in k: continue
            if k.endswith(".g"):
                k = k[:-1] + "weight"
            elif 'cross_attn' in k:
                if k.endswith('gamma'):
                    k = k[:-5] + "weight"
                elif k.endswith('beta'):
                    k = k[:-4] + "bias"
            assert k in decoder.state_dict().keys(), k
            flow_tensor = flow.tensor(torch_tensor.cpu().numpy(), placement=decoder.state_dict()[k].placement, sbp=decoder.state_dict()[k].sbp)
            decoder.state_dict()[k].data.copy_(flow_tensor.data)
        return decoder.eval()

    def load_pretrain_weight(
            self, 
            libai_cfg_model, 
            model_path
        ):
        # model = build_model(libai_cfg_model)
        # self.load_prior_weight(model.prior,libai_cfg_model.prior_weight_path)
        # self.load_decoder_weight(model.decoder, libai_cfg_model.decoder_weight_path)
        # return model
        return None
    
    def build_tokenizer(self, cfg):
        return instantiate(cfg.tokenizer)
        
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

    def __call__(self, text) -> dict:
        return self.forward(text)

    def forward(self, text) -> dict:
        tokens = self.tokenizer.tokenize(text).to_global(placement = flow.placement(type='cuda',ranks=[0,1,2,3]), sbp=flow.sbp.broadcast) 
        prior = self.load_prior_weight(self.cfg.model.prior, self.cfg.model.prior_weight_path)
        text_embed, text_encodings, text_mask = prior.clip.embed_text(tokens)
        image_embed = prior.sample(tokens, num_samples_per_batch = 2, cond_scale = 1.)
        del prior
        flow.cuda.empty_cache()

        decoder = self.load_decoder_weight(self.cfg.model.decoder, self.cfg.model.decoder_weight_path)
        image_embed = decoder.sample(image_embed = image_embed, text_encodings = text_encodings, text_mask = text_mask, cond_scale = 3.5)
        return {"image_embed": image_embed}

    def save_images(self, images):
        import flowvision.transforms as T
        to_pil = T.ToPILImage()
        images = images.to_local().to("cpu")
        images = list(map(to_pil, [images[i] for i in range(images.shape[0])]))
        for i,image in enumerate(images):
            image.save(f"./result_{i}.png")

if __name__ == "__main__":
    from libai.config import instantiate
    model = Dalle2Pipeline(
        "./dalle2_config.py", 
        data_parallel=1,
        tensor_parallel=4,
        pipeline_parallel=1)
    imgs = model(["a man is playing basketball with his friends!"] * 10)
    imgs = imgs['image_embed']
    rank = flow.env.get_rank()
    if rank == 0:
        imgs = imgs.to_global(placement=flow.placement(type='cuda', ranks=[0]), sbp=flow.sbp.broadcast)
        model.save_images(imgs)