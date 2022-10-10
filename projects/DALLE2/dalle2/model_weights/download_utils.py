import logging
import os

from libai.utils.file_utils import download_file

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

url_map = {
    "prior": "https://huggingface.co/nousr/conditioned-prior/resolve/main/vit-l-14/prior_aes_finetune.pth",  # noqa
    "decoder": "https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/decoder/1.5B_laion2B/latest.pth",  # noqa
    "clip": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",  # noqa
    "bpe_vocab": "https://oneflow-static.oss-cn-beijing.aliyuncs.com/libai/clip/bpe_simple_vocab_16e6.txt.gz",  # noqa
    "swinir": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/",
}


def _download_if_not_exist(path, name):
    if os.path.exists(path):
        logger.info((f"using {name}'s weight at {path}"))
        return
    if name == "swinir":
        os.makedirs(os.path.dirname(path), exist_ok=True)
        download_file(path, url_map[name] + os.path.basename(path))
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    download_file(path, url_map[name])


def download_dalle2_weights(cfg):
    if not os.path.exists("./dalle2/data/bpe_simple_vocab_16e6.txt.gz"):
        os.makedirs("./dalle2/data", exist_ok=True)
        download_file("./dalle2/data/bpe_simple_vocab_16e6.txt.gz", url_map["bpe_vocab"])

    _download_if_not_exist(cfg.swinir.swinir_path, "swinir")
    _download_if_not_exist(cfg.model.prior_weight_path, "prior")
    _download_if_not_exist(cfg.model.decoder_weight_path, "decoder")
    _download_if_not_exist(cfg.model.prior.clip.name, "clip")
