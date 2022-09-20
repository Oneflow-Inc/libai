from genericpath import exists
import os
from libai.utils.file_utils import download_file

    
    
def download_dalle2_weights():
    os.makedirs("./dalle2/model_weights", exist_ok=True)
    if not os.path.exists("./dalle2/model_weights/prior_aes_finetune.pth"):
        download_file("./dalle2/model_weights/", "https://huggingface.co/nousr/conditioned-prior/blob/main/vit-l-14/prior_aes_finetune.pth ")
    if not os.path.exists("./dalle2/model_weights/latest.pth"):
        download_file("./dalle2/model_weights/", "https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/decoder/1.5B_laion2B/latest.pth")
    if not os.path.exists("./dalle2/model_weights/ViT-L-14.pt"):
        download_file("./dalle2/model_weights/", "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt")
    if not os.path.exists("./dalle2/data/bpe_simple_vocab_16e6.txt"):
        download_file("./dalle2/data/", "https://github.com/lucidrains/DALLE2-pytorch/blob/main/dalle2_pytorch/data/bpe_simple_vocab_16e6.txt")