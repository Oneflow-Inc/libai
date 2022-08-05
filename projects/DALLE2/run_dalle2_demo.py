import oneflow as flow
import numpy as np
import torch
import os,sys
from download_utils import download_dalle2_weights
from dalle2 import tokenizer
from dalle2 import OpenAIClipAdapter, DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder 
from libai.layers import Embedding

def gen_text_and_img_emb(text = ['cute puppy chasing after a squirrel']):
    #clip use openai's ViT-L/14
    clip=OpenAIClipAdapter("./dalle2/model_weights/ViT-L-14.pt").eval()
    clip.to_global(placement=flow.placement(type='cuda',ranks=[0]), sbp=flow.sbp.broadcast)
    
    tokens = tokenizer.tokenize(text).to_global(placement = flow.placement(type='cuda',ranks=[0]), sbp=flow.sbp.broadcast) 
    _, text_encodings = clip.embed_text(tokens)
    np.save("text_encodings.npy", text_encodings.to_local().numpy())
    
    # prior networks (with transformer)
    prior_network = DiffusionPriorNetwork(
        dim = 768,
        depth = 24,
        num_timesteps = 1000,
        max_text_len = 77,
        num_time_embeds = 1,
        num_image_embeds=1,
        num_text_embeds = 1,
        dim_head = 64,
        heads = 32,
        ff_mult = 4,
        attn_dropout = 0.05,
        ff_dropout = 0.05,
        normformer = True,
    )
    
    diffusion_prior = DiffusionPrior(
        net = prior_network,
        clip = clip,
        image_embed_dim = 768,
        timesteps = 1000,
        cond_drop_prob = 0.1,
        loss_type="l2",
        condition_on_text_encodings = True
    )
    
    diffusion_prior.to_global(placement=flow.placement(type='cuda',ranks=[0]), sbp=flow.sbp.broadcast)
    #https://huggingface.co/nousr/conditioned-prior/blob/main/vit-l-14/prior_aes_finetune.pth
    state_dict = torch.load("./dalle2/model_weights/prior_aes_finetune.pth", map_location="cpu")['ema_model']
    for k,torch_tensor in state_dict.items():
        if "clip" in k:
            continue
        if k.endswith(".g"):
            k= k[:-1] + "weight" 
        elif k.startswith("net.causal_transformer"):
            if k.endswith("gamma"):
                k= k[:-5] + 'weight'
            elif k.endswith('beta'):
                k = k[:-4] + 'bias'
        assert k in diffusion_prior.state_dict(), k
        flow_tensor = flow.tensor(torch_tensor.cpu().numpy()).to(flow.float32).to_global(placement=flow.placement(type='cuda',ranks=[0]), sbp=flow.sbp.broadcast)
        diffusion_prior.state_dict()[k].data.copy_(flow_tensor.data)
    
    
    image_embed = diffusion_prior.sample(tokens, num_samples_per_batch = 2, cond_scale = 1.)
    np.save("image_embed.npy", image_embed.to_local().numpy())
    del clip, prior_network, diffusion_prior
    
def gen_images():
    image_embed = np.load("image_embed.npy")
    image_embed = flow.tensor(image_embed).to_global(placement=flow.placement(type='cuda',ranks=[0]), sbp=flow.sbp.broadcast)

    text_encodings = np.load("text_encodings.npy")[0:1]
    text_encodings = flow.tensor(text_encodings).to_global(placement=flow.placement(type='cuda',ranks=[0]), sbp=flow.sbp.broadcast)
    clip =None
    #https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/decoder/1.5B_laion2B/latest.pth
    #https://huggingface.co/laion/DALLE2-PyTorch/raw/main/decoder/1.5B_laion2B/decoder_config.json
    unet1 = Unet(
        dim = 320,
        image_embed_dim = 768,
        text_embed_dim = 768,
        cond_dim = 512,
        channels = 3,
        dim_mults=(1, 2, 3, 4),
        num_resnet_blocks = 4,
        attn_heads = 8,
        attn_dim_head = 64,
        sparse_attn  = True,
        memory_efficient = True,
        cond_on_text_encodings = True,    # set to True for any unets that need to be conditioned on text encodings
        self_attn = [False, True, True, True]
    ).eval()
    
    
    decoder = Decoder(
        unet = (unet1,),
        image_sizes = [64,],
        clip = clip,
        channels = 3,
        timesteps = 1000,
        loss_type = "l2",
        beta_schedule = ["cosine"],
        learned_variance = True
    ).eval()
    decoder.to_global(placement = flow.placement(type = 'cuda', ranks = [0]), sbp = flow.sbp.broadcast)
    state_dict = torch.load("./dalle2/model_weights/latest.pth", map_location = "cpu")
                
    for k, torch_tensor in state_dict.items():
        if 'clip' in k: continue
        if k.endswith(".g"):
            k = k[:-1] + "weight"
        elif 'cross_attn' in k:
            if k.endswith('gamma'):
                k = k[:-5] + weight
            elif k.endswith('beta'):
                k = k[:-4] + bias
        assert k in decoder.state_dict().keys(), k
        flow_tensor = flow.tensor(torch_tensor.cpu().numpy()).to(flow.float32).to_global(placement=flow.placement(type='cuda',ranks=[0]), sbp=flow.sbp.broadcast)
        decoder.state_dict()[k].data.copy_(flow_tensor.data)
    
    images = decoder.sample(image_embed = image_embed, text_encodings = text_encodings, cond_scale = 2.)
    return images

def save_images(images):
    import flowvision.transforms as T
    to_pil = T.ToPILImage()
    images = images.to_local().to("cpu")
    images = list(map(to_pil, [images[i] for i in range(images.shape[0])]))
    for i,image in enumerate(images):
        image.save(f"./result_{i}.png")

download_dalle2_weights()
gen_text_and_img_emb()
save_images(gen_images())
