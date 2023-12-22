# Detailed instruction on using model parallel in LiBai
This document is a tutorial for users to learn how to transfer a pytorch model to oneflow, and use model parallel in Libai for inference. We will first take the DALLE2 model for example, and then we will show how to use model parallel which can be easily done in libai.

**Note**: the code of DALLE2 is adapted from [this repo](https://github.com/lucidrains/DALLE2-pytorch), which is an unofficial implementation. The final result may differ from the original generated images in the [paper](https://arxiv.org/abs/2204.06125). You can also try the model in [google colab](https://colab.research.google.com/github/LAION-AI/dalle2-laion/blob/main/notebooks/dalle2_laion_alpha.ipynb).

## Transfer pytorch model to oneflow.
It's easy for user to transfer a pytorch model into oneflow, since most of oneflow's api is consistent with pytorch. First we change `import torch` to `import oneflow as flow`, and then we can replace all `torch` in the code to `flow`. If the model can work correctly in the originally
pytorch codes, it's likely to be able to work correctly in oneflow. Sometimes the program may raise error like 
```
AttributeError: module 'oneflow' has no attribute 'xxx'
```
try install the latest version of oneflow which might help, you can find more details [here](https://github.com/Oneflow-Inc/oneflow#install-oneflow).



**1、Download the pytorch DALLE2 model**:

As show in the [google colab](https://colab.research.google.com/github/LAION-AI/dalle2-laion/blob/main/notebooks/dalle2_laion_alpha.ipynb), we will use the version of 0.15.4, 
```
git clone -b v0.15.4 https://github.com/lucidrains/DALLE2-pytorch.git
```
the pretrained model weights can be found in huggingface: [the prior weight](https://huggingface.co/zenglishuci/conditioned-prior/resolve/main/vit-l-14/prior_aes_finetune.pth) and [the decoder weight](https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/decoder/1.5B_laion2B/latest.pth).
A simple inference script can be written as 
```python
# inference_dalle2.py
import numpy as np
import torch
import os,sys
from dalle2_pytorch import tokenizer
from dalle2_pytorch import OpenAIClipAdapter, DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder

def generate_images_from_text(texts):
    clip=OpenAIClipAdapter("ViT-L-14.pt").to("cuda")
    
    tokens = tokenizer.tokenize(text).to("cuda")
    _, text_encodings, text_mask = clip.embed_text(tokens)
    
    prior_network = DiffusionPriorNetwork(
        dim = 768,
        depth = 24,
        num_timesteps = 1000,
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
    state_dict = torch.load("prior_aes_finetune.pth", map_location="cpu")['ema_model']
    diffusion_prior.load_state_dict(state_dict, strict=True)
    diffusion_prior.to("cuda")

    image_embed = diffusion_prior.sample(tokens, num_samples_per_batch = 2, cond_scale = 1.)

    unet = Unet(
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
    )
    
    decoder = Decoder(
        unet = (unet,),
        image_sizes = [64],
        clip = clip,
        channels = 3,
        timesteps = 1000,
        loss_type = "l2",
        beta_schedule = ["cosine"],
        learned_variance = True
    )
    state_dict = torch.load("latest.pth", map_location = "cpu")
                
    new_dict = {}
    for k,v in state_dict.items():
        if 'clip.' in k: continue
        if ('cross_attn' in k or 'fn.fn.' in k) and k.endswith(".g"):
            k = k[:-1] + "gamma"
        new_dict[k] = v
        assert k in decoder.state_dict().keys(), k 
    decoder.load_state_dict(new_dict, strict=False)
    decoder.to("cuda")
    images = decoder.sample(image_embed = image_embed, text_encodings = text_encodings, text_mask = text_mask, cond_scale = 3.5)
    return images

def save_images(images):
    import torchvision.transforms as T
    to_pil = T.ToPILImage()
    images = list(map(to_pil,images.unbind(dim = 0))) 
    for i,image in enumerate(images):
        image.save(f"./result_{i}.png")

def main():
    text = ["a dolphin in an astronaut suit on saturn, artstation"]
    images = gen_text_and_img_emb(text)
    save_images(images)

if __name__ == "__main__":
    main()
```
run `python inference_dalle2.py`, this should work.



## 2、Change the deep learning framework to oneflow.
As mentioned above, we replace all the `torch` symbol to `flow` by firstly change `import torch` to `import oneflow as flow` in all python files. 
It should be noted that the original pytorch code also import other python packages using pytorch backend like [einops](https://github.com/arogozhnikov/einops)、[einops_ext](https://github.com/lucidrains/einops-exts)、[kornia](https://github.com/kornia/kornia) etc. which should also be modified at the same time.

Fortunately, only a few api of these packages are used, we can take out the relevant code from the github repos and merge them in a separate file.

For example, we can simply create the einops_ext.py file adapted from [here](https://github.com/lucidrains/einops-exts/blob/main/einops_exts/einops_exts.py), then we can import einops_ext from the python file which use oneflow instead of python packages using torch.
```python
# einops_ext.py
import re
from oneflow import nn #here change `from torch import nn` to `from oneflow import nn`
from functools import wraps, partial

from einops import rearrange, reduce, repeat
```



## 3、Using Libai's api. 
[LiBai](https://github.com/Oneflow-Inc/libai) is a large-scale open-source model training toolbox based on OneFlow.

Libai provides many efficient api which can be easily used for distributed training and evaluation. It also supports some popular models under the projects folder such as [CLIP](https://github.com/Oneflow-Inc/libai/tree/main/projects/CLIP). To avoid duplication of work, we directly use the clip model implemented in Libai. The relevant code in the original pytorch code is the `OpenAIClipAdapter` class which can be written as follows:
```python
# _clip.py
import os
import sys
import oneflow as flow
import oneflow.nn.functional as F
from oneflow import nn
from collections import namedtuple

def import_flow_clip(fn):
    
    def wrapper(*args, **kwargs):
        sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), "CLIP"))
        fn(*args, **kwargs)
        sys.path.pop()

    return wrapper

class BaseClipAdapter(nn.Module):
    pass

class OpenAIClipAdapter(BaseClipAdapter):

    @import_flow_clip
    def __init__(
        self,
        name = 'ViT-L/14'
    ):
        import clip
        openai_clip, preprocess = clip.load(name)
        super().__init__(openai_clip)

```

[DiffusionPrior](https://github.com/lucidrains/DALLE2-pytorch/blob/v0.15.4/dalle2_pytorch/dalle2_pytorch.py#L873) and [Decoder](https://github.com/lucidrains/DALLE2-pytorch/blob/v0.15.4/dalle2_pytorch/dalle2_pytorch.py#L1802) follow their original implementation.


**Using libai.layers** 

LiBai provides multiple parallelisms such as Data Parallelism, Tensor Parallelism, and Pipeline Parallelism. To experience these features, we will use libai.layers like Linear and LayerNorm:
```python
from libai.layers import Linear, LayerNorm
```
the nn.Linear will be replace with `libai.layers.Linear`.

**Compare the outputs** To make sure it is correctly modified from `torch` to `flow`,  it's necessary to compare the outputs to see if they are the same after the change. A notable point here is that in the sampling stage, the noise are randomly generated, like 
```python
noise = flow.randn(shape)
# or noise = torch.randn(shape) in torch code
``` 
torch and oneflow will generate different numbers here even if they are set the same random seed. An alternate way is to make a transition through numpy:
```python
import numpy as np
np.random.seed(6666)
noise = flow.tensor(np.randn(shape))
# or noise = torch.tensor(np.randn(shape)) in torch code 
```
When the model is fed the same input text, the output images by oneflow or torch code should be same.

**LazyConfig and LazyCall**

Oneflow provides LazyConfig system for more flexible syntax and no predefined structures, find more [here](https://libai.readthedocs.io/en/latest/tutorials/basics/Config_System.html). As for the DALLE2, the config file can be write as 
```python
from omegaconf import DictConfig
from libai.config import LazyCall
from dalle2.models import DiffusionPrior, DiffusionPriorNetwork, Unet, Decoder, DALLE2
from dalle2._clip import OpenAIClipAdapter

clip = LazyCall(OpenAIClipAdapter)(name="./dalle2/model_weights/ViT-L-14.pt")

prior = LazyCall(DiffusionPrior)(
    net = LazyCall(DiffusionPriorNetwork)(
        dim=768,
        depth=24,
        num_timesteps=1000,
        max_text_len=77,
        num_time_embeds=1,
        num_image_embeds=1,
        num_text_embeds=1,
        dim_head=64,
        heads=32,
        ff_mult=4,
        attn_dropout=0.05,
        ff_dropout=0.05,
        normformer=True,
    ),
    clip=clip,
    image_embed_dim=768,
    timesteps=1000,
    cond_drop_prob=0.1,
    loss_type="l2",
    condition_on_text_encodings=True
)

unet1 = LazyCall(Unet)(
    dim=320,
    image_embed_dim=768,
    text_embed_dim=768,
    cond_dim=512,
    channels=3,
    dim_mults=(1, 2, 3, 4),
    num_resnet_blocks=4,
    attn_heads=8,
    attn_dim_head=64,
    sparse_attn=True,
    memory_efficient=True,
    cond_on_text_encodings=True,    # set to True for any unets that need to be conditioned on text encodings
    self_attn=[False, True, True, True]
)

decoder = LazyCall(Decoder)(
    unet=(unet1,),
    image_sizes=[64, ],
    clip=None,
    channels=3,
    timesteps=1000,
    loss_type="l2",
    beta_schedule=["cosine"],
    learned_variance=True
)

dalle2_model = LazyCall(DALLE2)(
    prior=prior,
    decoder=decoder,
    prior_weight_path='',
    decoder_weight_path=''
)
```

## 4、Model parallel in Libai.
In order to achieve the model parallel inference under libai, we should set the parallel mode according to your needs. The default value of argument parallel is `data` in libai.layers.Linear, which means data parallel. To achieve model parallel, we need change the parallel to `col` or `row`. The most efficient way is to set the Linear layers in the col -> row -> col order.

A transformer block contains a attention and a feedforward submodule, and each submodule exactly contains 2 Linear layers. 
The attention module contains the qkv projection and out projection. Thus we set the qkv projection as `col`, and the out projection as `row`:
```python
#attention 
class Attention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # 1、 qkv projection
        self.to_q = Linear(dim, inner_dim, bias = False, parallel='col')
        self.to_kv = Linear(dim, dim_head * 2, bias = False, parallel='col') 
        #2、 output projection
        self.to_out = nn.Sequential(
            Linear(inner_dim, dim, bias = False, parallel='row'), #'row'
            LayerNorm(dim)
        )
```
and feed forward contains in projection and out projection, the former will be set `col` and the later will be set `row`.
```python
def FeedForward(
    dim,
    mult = 4,
    dropout = 0.,
    post_activation_norm = False
):
    inner_dim = int(mult * dim)
    return nn.Sequential(
        LayerNorm(dim),
        Linear(dim, inner_dim * 2, bias = False, parallel='col'), 
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout),
        Linear(inner_dim, dim, bias = False, parallel='row')
    )
```


for the single machine with 4 GPUs, the model parallel could be set like:
```python
import libai.utils.distributed as dist
dist.setup_dist_util(
    DictConfig(
        dict(
            data_parallel_size=1,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        )
    )
)
```



If you successfully complete the above steps, now you can have fun with the (unofficial) dalle2 model.