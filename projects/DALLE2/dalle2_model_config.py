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