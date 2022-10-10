import os
from typing import Dict

import oneflow as flow
from dalle2.dalle2_loader import Dalle2ModelLoader
from dalle2.model_weights.download_utils import download_dalle2_weights
from dalle2.tokenizer import SimpleTokenizer
from oneflow.framework import balanced_splitter

import libai.utils.distributed as dist
from libai.inference.basic import BasePipeline


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
        self.cfg.model.prior.clip.name = "./dalle2/model_weights/ViT-L-14.pt"
        self.cfg.model.prior_weight_path = "./dalle2/model_weights/prior_aes_finetune.pth"
        self.cfg.model.decoder_weight_path = "./dalle2/model_weights/latest.pth"
        self.cfg.swinir.swinir_path = (
            "./swinir/weights/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
        )

    def load_pretrain_weight(self, libai_cfg_model, model_path, mode=None):
        if dist.is_main_process():
            download_dalle2_weights(self.cfg)
        dist.synchronize()
        model_loader = Dalle2ModelLoader(libai_cfg_model, self.cfg, model_path)
        return model_loader.load()

    def build_tokenizer(self, cfg):
        return SimpleTokenizer()  # return instantiate(cfg.tokenizer)

    def _parse_parameters(self, model_path=None, save_images=False, upsample_scale=None, **kwargs):
        preprocess_params = {}
        forward_params = {
            "model_path": model_path,
            "num_samples_per_batch": kwargs.get("num_samples_per_batch", 2),
            "prior_cond_scale": kwargs.get("prior_cond_scale", 1.0),
            "decoder_cond_scale": kwargs.get("decoder_cond_scale", 3.5),
        }
        postprocess_params = {
            "save_images": save_images,
            "upsample_scale": upsample_scale,
            "swinir_path": "./swinir/weights/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        }

        return preprocess_params, forward_params, postprocess_params

    def split_data(self, text):
        rank = dist.get_rank()
        indices = balanced_splitter.BalancedRanges(len(text), dist.get_world_size())
        return text[indices[rank][0] : indices[rank][1]]

    def preprocess(self, input_, **preprocess_parameters: Dict) -> dict:
        tokens = self.tokenizer.tokenize(input_).to_global(
            placement=flow.placement(type="cuda", ranks=list(range(dist.get_world_size()))),
            sbp=flow.sbp.broadcast,
        )
        return {"text": input_, "tokens": tokens}

    def forward(self, model_input_dict, **forward_params) -> dict:
        tokens = model_input_dict["tokens"]
        text_embed, text_encodings, text_mask = self.model.prior.clip.embed_text(tokens)
        image_embed = self.model.prior.sample(
            tokens,
            num_samples_per_batch=forward_params["num_samples_per_batch"],
            cond_scale=forward_params["prior_cond_scale"],
        )

        image_embed = self.model.decoder.sample(
            image_embed=image_embed,
            text_encodings=text_encodings,
            text_mask=text_mask,
            cond_scale=forward_params["decoder_cond_scale"],
        )

        return {"image_embed": image_embed}

    def postprocess(self, model_output_dict, **postprocess_params: Dict) -> dict:
        if not postprocess_params.get("save_images", False):
            return model_output_dict
        output_path = postprocess_params.get("output_dit", "./outputs")
        os.makedirs(output_path, exist_ok=True)

        import flowvision.transforms as T

        to_pil = T.ToPILImage()
        images = model_output_dict["image_embed"].to("cpu")
        images_64x64 = list(map(to_pil, [images[i] for i in range(images.shape[0])]))
        for i, image in enumerate(images_64x64):
            image.save(f"{output_path}/{i}.png")
        if postprocess_params.get("upsample_scale", False):
            from swinir import load_model, upsample4x, upsample16x

            swinir = load_model(postprocess_params.get("swinir_path", ""))
            upsample_fun = upsample4x if args.upsample_scale == 4 else upsample16x
            images = upsample_fun(images, swinir).to("cpu")
            images = list(map(to_pil, [images[i] for i in range(images.shape[0])]))
            for i, image in enumerate(images):
                image.save(f"{output_path}/{i}_{args.upsample_scale}x.png")
        print(f"Images have been saved under {output_path}.")
        return model_output_dict


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/dalle2_config.py")
    parser.add_argument("--data_parallel", type=int, default=1)
    parser.add_argument("--tensor_parallel", type=int, default=4)
    parser.add_argument("--pipeline_parallel", type=int, default=1)
    parser.add_argument(
        "--upsample_scale",
        type=int,
        choices=[4, 16],
        default=None,
        help="upsample scale, if 4x, output resolution will be 256 x 256.",
    )
    parser.add_argument(
        "--swinir_path",
        type=str,
        default="./swinir/weights/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--save_images", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = Dalle2Pipeline(
        config_file=args.config_file,
        data_parallel=args.data_parallel,
        tensor_parallel=args.tensor_parallel,
        pipeline_parallel=args.pipeline_parallel,
    )

    texts = [
        "a shiba inu wearing a beret and black turtleneck",
        "a teddy bear on a skateboard in times square",
        "trump fight with biden in white house",
        "Donald trump fight with biden in white house",
    ]

    imgs = model(texts, **vars(args))
