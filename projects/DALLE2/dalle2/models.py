import math
import random
from contextlib import contextmanager
from functools import partial, wraps

import flowvision.transforms as T
import kornia.augmentation as K
import numpy as np
import oneflow as flow
import oneflow.nn.functional as F
from einops import rearrange, reduce, repeat
from kornia.filters import gaussian_blur2d
from omegaconf import ListConfig
from oneflow import einsum, nn
from oneflow.nn import Conv2d, ConvTranspose2d, GroupNorm
from oneflow.nn.functional import layer_norm
from resize_right import resize
from tqdm.auto import tqdm

from libai.layers import Embedding, LayerNorm, Linear
from libai.utils import distributed as dist

from .einops_exts import EinopsToAndFrom, Rearrange, check_shape, rearrange_many, repeat_many
from .rotary_embedding_flow import RotaryEmbedding
from .tokenizer import SimpleTokenizer
from .vqgan_vae import NullVQGanVAE, VQGanVAE

# rotary embeddings


# constants
def get_default_sbp():
    return dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])


def get_default_placement():
    return dist.get_layer_placement(0)


NAT = 1.0 / math.log(2.0)

random.seed(666)
np.random.seed(6666)
# helper functions


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)

    return inner


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(val, length=None):
    if isinstance(val, list) or isinstance(val, ListConfig):
        val = tuple(val)

    out = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(out) == length

    return out


def module_device(module):
    return next(module.parameters()).device


def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)


@contextmanager
def null_context(*args, **kwargs):
    yield


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


def pad_tuple_to_length(t, length, fillvalue=None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue,) * remain_length))


# for controlling freezing of CLIP


def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)


def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)


def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)


# tensor helpers


def log(t, eps=1e-12):
    return flow.log(t.clamp(min=eps))


def l2norm(t):
    return F.normalize(t, dim=-1)


def resize_image_to(image, target_image_size):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    scale_factors = target_image_size / orig_image_size
    return resize(image, scale_factors=scale_factors)


# image normalization functions
# ddpms expect images to be in the range of -1 to 1
# but CLIP may otherwise


def normalize_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5


# classifier free guidance functions


def prob_mask_like(shape, prob, placement=None, sbp=None):
    placement = placement or get_default_placement()
    sbp = sbp or get_default_sbp()
    if prob == 1:
        return flow.ones(shape, dtype=flow.bool, placement=placement, sbp=sbp)
    elif prob == 0:
        return flow.zeros(shape, dtype=flow.bool, placement=placement, sbp=sbp)
    else:
        return flow.zeros(shape, placement=placement, sbp=sbp).float().uniform_(0, 1) < prob


# gaussian diffusion helper functions


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(len(a.shape) - 1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def meanflat(x):
    return x.mean(dim=tuple(range(1, len(x.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + flow.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * flow.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + flow.tanh(((2.0 / math.pi) ** 0.5) * (x + 0.044715 * (x ** 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres=0.999):
    assert x.shape == means.shape == log_scales.shape

    # attempting to correct nan gradients when learned variance is turned on
    # in the setting of deepspeed fp16
    eps = 1e-12 if x.dtype == flow.float32 else 1e-3

    centered_x = x - means
    inv_stdv = flow.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus)
    log_one_minus_cdf_min = log(1.0 - cdf_min)
    cdf_delta = cdf_plus - cdf_min

    log_probs = flow.where(
        x < -thres, log_cdf_plus, flow.where(x > thres, log_one_minus_cdf_min, log(cdf_delta, eps))
    )

    return log_probs


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = flow.linspace(0, timesteps, steps, dtype=flow.float64)
    alphas_cumprod = flow.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / first(alphas_cumprod)
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return flow.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return flow.linspace(beta_start, beta_end, timesteps, dtype=flow.float64)


def quadratic_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return flow.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps, dtype=flow.float64) ** 2


def sigmoid_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = flow.linspace(-6, 6, timesteps, dtype=flow.float64)
    return flow.sigmoid(betas) * (beta_end - beta_start) + beta_start


class NoiseScheduler(nn.Module):
    def __init__(
        self, *, beta_schedule, timesteps, loss_type, p2_loss_weight_gamma=0.0, p2_loss_weight_k=1
    ):
        super().__init__()

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == "jsd":
            betas = 1.0 / flow.linspace(timesteps, 1, timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise NotImplementedError()

        betas = betas
        alphas = 1.0 - betas
        alphas_cumprod = flow.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        if loss_type == "l1":
            loss_fn = F.l1_loss
        elif loss_type == "l2":
            loss_fn = flow.nn.MSELoss()
        elif loss_type == "huber":
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # register buffer helper function to cast double back to float

        def register_buffer(name, val):
            self.register_buffer(name, val.to(flow.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", flow.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", flow.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", flow.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", flow.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", flow.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        register_buffer(
            "posterior_log_variance_clipped", flow.log(posterior_variance.clamp(min=1e-20))
        )
        register_buffer(
            "posterior_mean_coef1", betas * flow.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * flow.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # p2 loss reweighting

        self.has_p2_loss_reweighting = p2_loss_weight_gamma > 0.0
        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma,
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: flow.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p2_reweigh_loss(self, loss, times):
        if not self.has_p2_loss_reweighting:
            return loss
        return loss * extract(self.p2_loss_weight, times, loss.shape)


# diffusion prior


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(flow.ones(1, dim, 1, 1))

    def forward(self, x):
        # var = flow.var(x, dim = 1, unbiased = False, keepdim = True)
        # mean = flow.mean(x, dim = 1, keepdim = True)
        # return (x - mean) / (var + self.eps).sqrt() * self.weight
        x = x.permute(0, 2, 3, 1)
        out = layer_norm(x, normalized_shape=(x.shape[-1:]), eps=self.eps)
        return out.permute(0, 3, 1, 2) * self.weight


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# mlp


class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        *,
        expansion_factor=2.0,
        depth=2,
        norm=False,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)

        def norm_fn():
            return LayerNorm(hidden_dim) if norm else nn.Identity()

        layers = [nn.Sequential(Linear(dim_in, hidden_dim), nn.SiLU(), norm_fn())]

        for _ in range(depth - 1):
            layers.append(nn.Sequential(Linear(hidden_dim, hidden_dim), nn.SiLU(), norm_fn()))

        layers.append(Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


# relative positional bias for causal transformer


class RelPosBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        n = -relative_position
        n = flow.max(n, flow.zeros_like(n)).long()

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                flow.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = flow.min(val_if_large, flow.zeros_like(val_if_large) + num_buckets - 1)
        return flow.where(is_small, n, val_if_large)

    def forward(self, i, j):
        q_pos = flow.arange(i, dtype=flow.long)
        k_pos = flow.arange(j, dtype=flow.long)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(
            rp_bucket.to_global(sbp=get_default_sbp(), placement=get_default_placement())
        )
        return rearrange(values, "i j h -> h i j")


# feedforward


class SwiGLU(nn.Module):
    """used successfully in https://arxiv.org/abs/2204.0231"""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


def FeedForward(dim, mult=4, dropout=0.0, post_activation_norm=False):
    """post-activation norm https://arxiv.org/abs/2110.09456"""

    inner_dim = int(mult * dim)
    return nn.Sequential(
        LayerNorm(dim),
        Linear(dim, inner_dim * 2, bias=False, parallel="col"),
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout),
        Linear(inner_dim, dim, bias=False, parallel="row"),
    )


# attention


class Attention(nn.Module):
    def __init__(self, dim, *, dim_head=64, heads=8, dropout=0.0, causal=False, rotary_emb=None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(flow.randn(2, dim_head))
        self.to_q = Linear(dim, inner_dim, bias=False, parallel="col")
        self.to_kv = Linear(dim, dim_head * 2, bias=False, parallel="col")

        self.rotary_emb = rotary_emb

        self.to_out = nn.Sequential(
            Linear(inner_dim, dim, bias=False, parallel="row"), LayerNorm(dim)
        )

    def forward(self, x, mask=None, attn_bias=None):
        b, n = x.shape[:2]

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        q = q * self.scale

        # rotary embeddings

        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), "d -> b 1 d", b=b)

        k = flow.cat((nk, k), dim=-2)
        v = flow.cat((nv, v), dim=-2)

        # calculate query / key similarities
        sim = einsum("b h i d, b j d -> b h i j", q, k)
        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -3.4028e38  # flow.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=1)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(1 - mask, max_neg_value)  # ~mask
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = flow.ones(
                (i, j), placement=get_default_placement(), sbp=get_default_sbp(), dtype=flow.int32
            ).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)
        # attention

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        norm_out=True,
        attn_dropout=0.0,
        ff_dropout=0.0,
        final_proj=True,
        normformer=False,
        rotary_emb=True,
    ):
        super().__init__()
        self.rel_pos_bias = RelPosBias(heads=heads)

        rotary_emb = RotaryEmbedding(dim=min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            causal=True,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_emb=rotary_emb,
                        ),
                        FeedForward(
                            dim=dim,
                            mult=ff_mult,
                            dropout=ff_dropout,
                            post_activation_norm=normformer,
                        ),
                    ]
                )
            )
        self.norm = LayerNorm(dim) if norm_out else nn.Identity()
        self.project_out = Linear(dim, dim, bias=False) if final_proj else nn.Identity()

    def forward(
        self,
        x,
        mask=None,
    ):
        n = x.shape[1]

        attn_bias = self.rel_pos_bias(n, n + 1)

        for attn, ff in self.layers:
            x = attn(x, mask=mask, attn_bias=attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)


class DiffusionPriorNetwork(nn.Module):
    def __init__(
        self,
        dim,
        num_timesteps=None,
        num_time_embeds=1,
        num_image_embeds=1,
        num_text_embeds=1,
        max_text_len=256,
        **kwargs,
    ):
        super().__init__()
        self.num_time_embeds = num_time_embeds
        self.num_image_embeds = num_image_embeds
        self.num_text_embeds = num_text_embeds

        self.to_text_embeds = nn.Sequential(
            Linear(dim, dim * num_text_embeds) if num_text_embeds > 1 else nn.Identity(),
            Rearrange("b (n d) -> b n d", n=num_text_embeds),
        )

        self.to_time_embeds = nn.Sequential(
            Embedding(num_timesteps, dim * num_time_embeds)
            if exists(num_timesteps)
            else nn.Sequential(
                SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)
            ),  # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange("b (n d) -> b n d", n=num_time_embeds),
        )

        self.to_image_embeds = nn.Sequential(
            Linear(dim, dim * num_image_embeds) if num_image_embeds > 1 else nn.Identity(),
            Rearrange("b (n d) -> b n d", n=num_image_embeds),
        )

        self.learned_query = nn.Parameter(flow.randn(dim))
        self.causal_transformer = CausalTransformer(dim=dim, **kwargs)

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        text_embed,
        text_encodings=None,
        mask=None,
        cond_drop_prob=0.0,
    ):
        batch, dim, dtype = *image_embed.shape, image_embed.dtype

        num_time_embeds, num_image_embeds, num_text_embeds = (
            self.num_time_embeds,
            self.num_image_embeds,
            self.num_text_embeds,
        )

        text_embed = self.to_text_embeds(text_embed)
        image_embed = self.to_image_embeds(image_embed)

        # make text encodings optional
        # although the paper seems to suggest it is present <--

        if not exists(text_encodings):
            text_encodings = flow.empty(
                (batch, 0, dim),
                placement=get_default_placement(),
                sbp=get_default_sbp(),
                dtype=dtype,
            )

        if not exists(mask):
            mask = flow.ones(
                (batch, text_encodings.shape[-2]),
                placement=get_default_placement(),
                sbp=get_default_sbp(),
                dtype=flow.bool,
            )

        # classifier free guidance

        keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob)
        keep_mask = rearrange(keep_mask, "b -> b 1").to_global(
            placement=get_default_placement(), sbp=get_default_sbp()
        )

        mask &= keep_mask

        keep_mask = repeat(keep_mask, "b 1 -> b n", n=num_text_embeds)
        mask = flow.cat((mask, keep_mask), dim=1)

        if exists(mask):
            attend_padding = (
                1 + num_time_embeds + num_image_embeds
            )  # 1 for learned queries + number of image embeds + time embeds
            mask = F.pad(mask.to(flow.int32), (0, attend_padding), value=1)

        time_embed = self.to_time_embeds(
            diffusion_timesteps.to_global(placement=get_default_placement(), sbp=get_default_sbp())
        )

        learned_queries = repeat(self.learned_query, "d -> b 1 d", b=batch)

        tokens = flow.cat(
            (text_encodings, text_embed, time_embed, image_embed, learned_queries), dim=-2
        )

        # attend

        tokens = self.causal_transformer(tokens, mask=mask)

        # get learned query, which should predict the image embedding (per DDPM timestep)

        pred_image_embed = tokens[..., -1, :]

        return pred_image_embed


class DiffusionPrior(nn.Module):
    def __init__(
        self,
        net,
        *,
        clip=None,
        image_embed_dim=None,
        image_size=None,
        image_channels=3,
        timesteps=1000,
        cond_drop_prob=0.0,
        loss_type="l2",
        predict_x_start=True,
        beta_schedule="cosine",
        condition_on_text_encodings=True,
        sampling_clamp_l2norm=False,
        training_clamp_l2norm=False,
        init_image_embed_l2norm=False,
        image_embed_scale=None,
        clip_adapter_overrides=dict(),
    ):
        super().__init__()

        self.noise_scheduler = NoiseScheduler(
            beta_schedule=beta_schedule, timesteps=timesteps, loss_type=loss_type
        )

        if exists(clip):
            assert (
                image_channels == clip.image_channels
            ), f"channels of image ({image_channels}) should be equal to the channels "
            "that CLIP accepts ({clip.image_channels})"

            freeze_model_and_make_eval_(clip)
            self.clip = clip
        else:
            assert exists(
                image_embed_dim
            ), "latent dimension must be given, if training prior network without CLIP given"
            self.clip = None

        self.net = net
        self.image_embed_dim = default(image_embed_dim, lambda: clip.dim_latent)
        self.channels = default(image_channels, lambda: clip.image_channels)

        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.0
        self.condition_on_text_encodings = condition_on_text_encodings

        self.predict_x_start = predict_x_start

        self.image_embed_scale = default(image_embed_scale, self.image_embed_dim ** 0.5)

        # whether to force an l2norm, similar to clipping denoised, when sampling
        self.sampling_clamp_l2norm = sampling_clamp_l2norm
        self.training_clamp_l2norm = training_clamp_l2norm
        self.init_image_embed_l2norm = init_image_embed_l2norm

        # device tracker
        self.register_buffer("_dummy", flow.tensor([True]), persistent=False)

    def p_mean_variance(self, x, t, text_cond, clip_denoised=False, cond_scale=1.0):
        assert not (
            cond_scale != 1.0 and not self.can_classifier_guidance
        ), "the model was not trained with conditional dropout, "
        "and thus one cannot use classifier free guidance (cond_scale anything other than 1)"

        pred = self.net.forward_with_cond_scale(x, t, cond_scale=cond_scale, **text_cond)

        if self.predict_x_start:
            x_recon = pred
        else:
            x_recon = self.noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        if clip_denoised and not self.predict_x_start:
            x_recon.clamp_(-1.0, 1.0)

        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_recon = l2norm(x_recon) * self.image_embed_scale

        model_mean, posterior_variance, posterior_log_variance = self.noise_scheduler.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @flow.no_grad()
    def p_sample(self, x, t, text_cond=None, clip_denoised=True, cond_scale=1.0):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, text_cond=text_cond, clip_denoised=clip_denoised, cond_scale=cond_scale
        )
        noise = flow.randn(*x.shape, placement=get_default_placement(), sbp=get_default_sbp())
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @flow.no_grad()
    def p_sample_loop(self, shape, text_cond, cond_scale=1.0):
        b = shape[0]
        image_embed = flow.randn(*shape, placement=get_default_placement(), sbp=get_default_sbp())

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(
            reversed(range(0, self.noise_scheduler.num_timesteps)),
            desc="sampling loop time step",
            total=self.noise_scheduler.num_timesteps,
        ):
            times = flow.full(
                (b,), i, placement=get_default_placement(), sbp=get_default_sbp(), dtype=flow.long
            )
            image_embed = self.p_sample(
                image_embed, times, text_cond=text_cond, cond_scale=cond_scale
            )

        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise=None):
        noise = default(
            noise,
            lambda: flow.randn(
                *image_embed.shape, placement=get_default_placement(), sbp=get_default_sbp()
            ),
        )

        image_embed_noisy = self.noise_scheduler.q_sample(x_start=image_embed, t=times, noise=noise)

        pred = self.net(image_embed_noisy, times, cond_drop_prob=self.cond_drop_prob, **text_cond)

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = l2norm(pred) * self.image_embed_scale

        target = noise if not self.predict_x_start else image_embed

        loss = self.noise_scheduler.loss_fn(pred, target)
        return loss

    @flow.no_grad()
    @eval_decorator
    def sample_batch_size(self, batch_size, text_cond, cond_scale=1.0):
        shape = (batch_size, self.image_embed_dim)

        img = flow.randn(*shape, placement=get_default_placement(), sbp=get_default_sbp())

        for i in tqdm(
            reversed(range(0, self.noise_scheduler.num_timesteps)),
            desc="sampling loop time step",
            total=self.noise_scheduler.num_timesteps,
        ):
            img = self.p_sample(
                img,
                flow.full(
                    (batch_size,),
                    i,
                    placement=get_default_placement(),
                    sbp=get_default_sbp(),
                    dtype=flow.long,
                ),
                text_cond=text_cond,
                cond_scale=cond_scale,
            )
        return img

    @flow.no_grad()
    @eval_decorator
    def sample(
        self,
        text,
        num_samples_per_batch=2,
        cond_scale=1.0,
        text_embed=None,
        text_encodings=None,
        text_mask=None,
    ):
        # in the paper, what they did was
        # sample 2 image embeddings, choose the top 1 similarity, as judged by CLIP
        text = repeat(text, "b ... -> (b r) ...", r=num_samples_per_batch)

        batch_size = text.shape[0]
        image_embed_dim = self.image_embed_dim

        if text_embed is None:
            assert self.clip is not None
            text_embed, text_encodings, text_mask = self.clip.embed_text(text)

        text_cond = dict(text_embed=text_embed)

        if self.condition_on_text_encodings:
            text_cond = {**text_cond, "text_encodings": text_encodings, "mask": text_mask}

        image_embeds = self.p_sample_loop(
            (batch_size, image_embed_dim), text_cond=text_cond, cond_scale=cond_scale
        )

        # retrieve original unscaled image embed

        image_embeds /= self.image_embed_scale

        text_embeds = text_cond["text_embed"]

        text_embeds = rearrange(text_embeds, "(b r) d -> b r d", r=num_samples_per_batch)
        image_embeds = rearrange(image_embeds, "(b r) d -> b r d", r=num_samples_per_batch)

        text_image_sims = einsum("b r d, b r d -> b r", l2norm(text_embeds), l2norm(image_embeds))
        top_sim_indices = text_image_sims.topk(k=1)[1]  # .indices

        top_sim_indices = repeat(top_sim_indices, "b 1 -> b 1 d", d=image_embed_dim)

        top_image_embeds = image_embeds.gather(1, top_sim_indices)
        return rearrange(top_image_embeds, "b 1 d -> b d")

    def forward(
        self,
        text=None,
        image=None,
        text_embed=None,  # allow for training on preprocessed CLIP text and image embeddings
        image_embed=None,
        text_encodings=None,  # as well as CLIP text encodings
        text_mask=None,
        *args,
        **kwargs,
    ):
        assert exists(text) ^ exists(text_embed), "either text or text embedding must be supplied"
        assert exists(image) ^ exists(image_embed), "either text or text embedding must be supplied"
        assert not (
            self.condition_on_text_encodings and (not exists(text_encodings) and not exists(text))
        ), "text encodings must be present if you specified to condition on it on initialization"

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # calculate text conditionings, based on what is passed in

        if exists(text):
            text_embed, text_encodings, text_mask = self.clip.embed_text(text)

        text_cond = dict(text_embed=text_embed)

        if self.condition_on_text_encodings:
            assert exists(
                text_encodings
            ), "text encodings must be present for diffusion prior if specified"
            text_cond = {**text_cond, "text_encodings": text_encodings, "mask": text_mask}

        # timestep conditioning from ddpm

        batch = image_embed.shape[0]
        times = flow.randint(
            0,
            self.noise_scheduler.num_timesteps,
            (batch,),
            placement=get_default_placement(),
            sbp=get_default_sbp(),
            dtype=flow.long,
        )

        # scale image embed (Katherine)

        image_embed *= self.image_embed_scale

        # calculate forward loss

        return self.p_losses(image_embed, times, text_cond=text_cond, *args, **kwargs)


# decoder


def ConvTransposeUpsample(dim, dim_out=None):
    dim_out = default(dim_out, dim)
    return ConvTranspose2d(dim, dim_out, 4, 2, 1)


def NearestUpsample(dim, dim_out=None):
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"), Conv2d(dim, dim_out, 3, padding=1)
    )


def Downsample(dim, *, dim_out=None):
    dim_out = default(dim_out, dim)
    return Conv2d(dim, dim_out, 4, 2, 1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = flow.exp(
            flow.arange(half_dim, placement=get_default_placement(), sbp=get_default_sbp()) * -emb
        )
        emb = rearrange(x, "i -> i 1") * rearrange(emb, "j -> 1 j")
        return flow.cat((emb.sin(), emb.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.project = Conv2d(dim, dim_out, 3, padding=1)
        self.norm = GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.project(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, cond_dim=None, time_cond_dim=None, groups=8):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(nn.SiLU(), Linear(time_cond_dim, dim_out * 2))

        self.cross_attn = None

        if exists(cond_dim):
            self.cross_attn = EinopsToAndFrom(
                "b c h w", "b (h w) c", CrossAttention(dim=dim_out, context_dim=cond_dim)
            )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        if exists(self.cross_attn):
            assert exists(cond)
            h = self.cross_attn(h, context=cond) + h

        h = self.block2(h)
        return h + self.res_conv(x)


class CrossAttention(nn.Module):
    def __init__(
        self, dim, *, context_dim=None, dim_head=64, heads=8, dropout=0.0, norm_context=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(
            flow.randn(2, dim_head, placement=get_default_placement(), sbp=get_default_sbp())
        )
        self.to_q = Linear(dim, inner_dim, bias=False, parallel="col")
        self.to_kv = Linear(context_dim, inner_dim * 2, bias=False, parallel="col")

        self.to_out = nn.Sequential(
            Linear(inner_dim, dim, bias=False, parallel="row"), LayerNorm(dim)
        )

    def forward(self, x, context, mask=None):
        b, n = x.shape[:2]

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), "d -> b h 1 d", h=self.heads, b=b)

        k = flow.cat((nk, k), dim=-2)
        v = flow.cat((nv, v), dim=-2)

        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        max_neg_value = -3.4028e38  # -flow.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=1)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(1 - mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head=32, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.GELU()
        self.to_qkv = Conv2d(dim, inner_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(Conv2d(inner_dim, dim, 1, bias=False), ChanLayerNorm(dim))

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = self.to_qkv(fmap).chunk(3, dim=1)
        q, k, v = rearrange_many((q, k, v), "b (h c) x y -> (b h) (x y) c", h=h)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum("b n d, b n e -> b d e", k, v)
        out = einsum("b n d, b d e -> b n e", q, context)
        out = rearrange(out, "(b h) (x y) d -> b (h d) x y", h=h, x=x, y=y)

        out = self.nonlin(out)
        return self.to_out(out)


class CrossEmbedLayer(nn.Module):
    def __init__(self, dim_in, kernel_sizes, dim_out=None, stride=2):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                Conv2d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2)
            )

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return flow.cat(fmaps, dim=1)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        *,
        image_embed_dim=None,
        text_embed_dim=None,
        cond_dim=None,
        num_image_tokens=4,
        num_time_tokens=2,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        channels_out=None,
        self_attn=False,
        attn_dim_head=32,
        attn_heads=16,
        lowres_cond=False,  #
        sparse_attn=False,
        attend_at_middle=True,
        cond_on_text_encodings=False,
        max_text_len=256,
        cond_on_image_embeds=False,
        add_image_embeds_to_time=True,  #
        init_dim=None,
        init_conv_kernel_size=7,
        resnet_groups=8,
        num_resnet_blocks=2,
        init_cross_embed_kernel_sizes=(3, 7, 15),
        cross_embed_downsample=False,
        cross_embed_downsample_kernel_sizes=(2, 4),
        memory_efficient=False,
        scale_skip_connection=False,
        nearest_upsample=False,
        final_conv_kernel_size=1,
        **kwargs,
    ):
        super().__init__()
        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        del self._locals["self"]
        del self._locals["__class__"]

        # for eventual cascading diffusion

        self.lowres_cond = lowres_cond

        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        init_channels = channels if not lowres_cond else channels * 2
        init_dim = default(init_dim, dim)

        self.init_conv = CrossEmbedLayer(
            init_channels, dim_out=init_dim, kernel_sizes=init_cross_embed_kernel_sizes, stride=1
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        num_stages = len(in_out)

        # time, image embeddings, and optional text encoding

        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4

        self.to_time_hiddens = nn.Sequential(
            SinusoidalPosEmb(dim), Linear(dim, time_cond_dim, parallel="col"), nn.GELU()
        )

        self.to_time_tokens = nn.Sequential(
            Linear(time_cond_dim, cond_dim * num_time_tokens, parallel="row"),
            Rearrange("b (r d) -> b r d", r=num_time_tokens),
        )

        self.to_time_cond = nn.Sequential(Linear(time_cond_dim, time_cond_dim, parallel="row"))

        self.image_to_tokens = (
            nn.Sequential(
                Linear(image_embed_dim, cond_dim * num_image_tokens),
                Rearrange("b (n d) -> b n d", n=num_image_tokens),
            )
            if cond_on_image_embeds and image_embed_dim != cond_dim
            else nn.Identity()
        )

        self.to_image_hiddens = (
            nn.Sequential(Linear(image_embed_dim, time_cond_dim), nn.GELU())
            if cond_on_image_embeds and add_image_embeds_to_time
            else None
        )

        self.norm_cond = LayerNorm(cond_dim)
        self.norm_mid_cond = LayerNorm(cond_dim)

        # text encoding conditioning (optional)

        self.text_to_cond = None

        if cond_on_text_encodings:
            assert exists(
                text_embed_dim
            ), "text_embed_dim must be given to the unet if cond_on_text_encodings is True"
            self.text_to_cond = Linear(text_embed_dim, cond_dim)

        # finer control over whether to condition on image embeddings and text encodings
        # so one can have the latter unets in the cascading DDPMs only focus on super-resoluting

        self.cond_on_text_encodings = cond_on_text_encodings
        self.cond_on_image_embeds = cond_on_image_embeds

        # for classifier free guidance

        self.null_image_embed = nn.Parameter(flow.randn(1, num_image_tokens, cond_dim))
        self.null_image_hiddens = nn.Parameter(flow.randn(1, time_cond_dim))

        self.max_text_len = max_text_len
        self.null_text_embed = nn.Parameter(flow.randn(1, max_text_len, cond_dim))

        # whether to scale skip connection, adopted in Imagen

        self.skip_connect_scale = 1.0 if not scale_skip_connection else (2 ** -0.5)

        # attention related params

        attn_kwargs = dict(heads=attn_heads, dim_head=attn_dim_head)

        self_attn = cast_tuple(self_attn, num_stages)

        def create_self_attn(dim):
            return EinopsToAndFrom("b c h w", "b (h w) c", Residual(Attention(dim, **attn_kwargs)))

        # resnet block klass

        resnet_groups = cast_tuple(resnet_groups, num_stages)
        top_level_resnet_group = first(resnet_groups)

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_stages)

        # downsample klass

        downsample_klass = Downsample
        if cross_embed_downsample:
            downsample_klass = partial(
                CrossEmbedLayer, kernel_sizes=cross_embed_downsample_kernel_sizes
            )

        # upsample klass

        upsample_klass = ConvTransposeUpsample if not nearest_upsample else NearestUpsample

        # give memory efficient unet an initial resnet block

        self.init_resnet_block = (
            ResnetBlock(
                init_dim, init_dim, time_cond_dim=time_cond_dim, groups=top_level_resnet_group
            )
            if memory_efficient
            else None
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        skip_connect_dims = []  # keeping track of skip connection dimensions

        for ind, ((dim_in, dim_out), groups, layer_num_resnet_blocks, layer_self_attn) in enumerate(
            zip(in_out, resnet_groups, num_resnet_blocks, self_attn)
        ):
            is_first = ind == 0
            is_last = ind >= (num_resolutions - 1)
            layer_cond_dim = cond_dim if not is_first else None

            dim_layer = dim_out if memory_efficient else dim_in
            skip_connect_dims.append(dim_layer)

            attention = nn.Identity()
            if layer_self_attn:
                attention = create_self_attn(dim_layer)
            elif sparse_attn:
                attention = Residual(LinearAttention(dim_layer, **attn_kwargs))

            self.downs.append(
                nn.ModuleList(
                    [
                        downsample_klass(dim_in, dim_out=dim_out) if memory_efficient else None,
                        ResnetBlock(
                            dim_layer, dim_layer, time_cond_dim=time_cond_dim, groups=groups
                        ),
                        nn.ModuleList(
                            [
                                ResnetBlock(
                                    dim_layer,
                                    dim_layer,
                                    cond_dim=layer_cond_dim,
                                    time_cond_dim=time_cond_dim,
                                    groups=groups,
                                )
                                for _ in range(layer_num_resnet_blocks)
                            ]
                        ),
                        attention,
                        downsample_klass(dim_layer, dim_out=dim_out)
                        if not is_last and not memory_efficient
                        else Conv2d(dim_layer, dim_out, 1),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[-1],
        )
        self.mid_attn = create_self_attn(mid_dim)
        self.mid_block2 = ResnetBlock(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[-1],
        )

        for ind, ((dim_in, dim_out), groups, layer_num_resnet_blocks, layer_self_attn) in enumerate(
            zip(
                reversed(in_out),
                reversed(resnet_groups),
                reversed(num_resnet_blocks),
                reversed(self_attn),
            )
        ):
            is_last = ind >= (len(in_out) - 1)
            layer_cond_dim = cond_dim if not is_last else None

            skip_connect_dim = skip_connect_dims.pop()

            attention = nn.Identity()
            if layer_self_attn:
                attention = create_self_attn(dim_out)
            elif sparse_attn:
                attention = Residual(LinearAttention(dim_out, **attn_kwargs))

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_out + skip_connect_dim,
                            dim_out,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=groups,
                        ),
                        nn.ModuleList(
                            [
                                ResnetBlock(
                                    dim_out + skip_connect_dim,
                                    dim_out,
                                    cond_dim=layer_cond_dim,
                                    time_cond_dim=time_cond_dim,
                                    groups=groups,
                                )
                                for _ in range(layer_num_resnet_blocks)
                            ]
                        ),
                        attention,
                        upsample_klass(dim_out, dim_in)
                        if not is_last or memory_efficient
                        else nn.Identity(),
                    ]
                )
            )

        self.final_resnet_block = ResnetBlock(
            dim * 2, dim, time_cond_dim=time_cond_dim, groups=top_level_resnet_group
        )
        self.to_out = Conv2d(
            dim,
            self.channels_out,
            kernel_size=final_conv_kernel_size,
            padding=final_conv_kernel_size // 2,
        )

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self, *, lowres_cond, channels, channels_out, cond_on_image_embeds, cond_on_text_encodings
    ):
        if (
            lowres_cond == self.lowres_cond
            and channels == self.channels
            and cond_on_image_embeds == self.cond_on_image_embeds
            and cond_on_text_encodings == self.cond_on_text_encodings
            and channels_out == self.channels_out
        ):
            return self

        updated_kwargs = dict(
            lowres_cond=lowres_cond,
            channels=channels,
            channels_out=channels_out,
            cond_on_image_embeds=cond_on_image_embeds,
            cond_on_text_encodings=cond_on_text_encodings,
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(
            *args, text_cond_drop_prob=1.0, image_cond_drop_prob=1.0, **kwargs
        )
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        *,
        image_embed,
        lowres_cond_img=None,
        text_encodings=None,
        text_mask=None,
        image_cond_drop_prob=0.0,
        text_cond_drop_prob=0.0,
        blur_sigma=None,
        blur_kernel_size=None,
    ):
        batch_size = x.shape[0]

        # add low resolution conditioning, if present

        assert not (
            self.lowres_cond and not exists(lowres_cond_img)
        ), "low resolution conditioning image must be present"

        if exists(lowres_cond_img):
            x = flow.cat((x, lowres_cond_img), dim=1)

        # initial convolution

        x = self.init_conv(x)
        r = x.clone()  # final residual
        # time conditioning

        time_hiddens = self.to_time_hiddens(time)

        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        # conditional dropout

        image_keep_mask = prob_mask_like((batch_size,), 1 - image_cond_drop_prob)
        text_keep_mask = prob_mask_like((batch_size,), 1 - text_cond_drop_prob)

        text_keep_mask = rearrange(text_keep_mask, "b -> b 1 1")

        # image embedding to be summed to time embedding
        # discovered by @mhh0318 in the paper

        if exists(image_embed) and exists(self.to_image_hiddens):
            image_hiddens = self.to_image_hiddens(image_embed)
            image_keep_mask_hidden = rearrange(image_keep_mask, "b -> b 1")
            null_image_hiddens = self.null_image_hiddens.to(image_hiddens.dtype)

            image_hiddens = flow.where(image_keep_mask_hidden, image_hiddens, null_image_hiddens)

            t = t + image_hiddens

        # mask out image embedding depending on condition dropout
        # for classifier free guidance

        image_tokens = None

        if self.cond_on_image_embeds:
            image_keep_mask_embed = rearrange(image_keep_mask, "b -> b 1 1")
            image_tokens = self.image_to_tokens(image_embed)
            null_image_embed = self.null_image_embed.to(
                image_tokens.dtype
            )  # for some reason pyflow AMP not working

            image_tokens = flow.where(image_keep_mask_embed, image_tokens, null_image_embed)

        # take care of text encodings (optional)

        text_tokens = None

        if exists(text_encodings) and self.cond_on_text_encodings:
            text_tokens = self.text_to_cond(text_encodings)
            text_tokens = text_tokens[:, : self.max_text_len]

            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len

            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

            if exists(text_mask):
                if remainder > 0:
                    # text_mask = F.pad(text_mask, (0, remainder), value = False)
                    text_mask = F.pad(text_mask.to(flow.int32), (0, remainder), value=0)

                text_mask = rearrange(text_mask, "b n -> b n 1")
                text_keep_mask = text_mask & text_keep_mask

            null_text_embed = self.null_text_embed.to(
                text_tokens.dtype
            )  # for some reason pyflow AMP not working

            text_tokens = flow.where(text_keep_mask, text_tokens, null_text_embed)

        # main conditioning tokens (c)
        c = time_tokens

        if exists(image_tokens):
            c = flow.cat((c, image_tokens), dim=-2)

        # text and image conditioning tokens (mid_c), to save on compute,
        # only do cross attention based conditioning on the inner most layers of the Unet

        mid_c = c if not exists(text_tokens) else flow.cat((c, text_tokens), dim=-2)

        # normalize conditioning tokens
        c = self.norm_cond(c)
        mid_c = self.norm_mid_cond(mid_c)
        # initial resnet block

        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)
        # go through the layers of the unet, down and up

        hiddens = []
        for pre_downsample, init_block, resnet_blocks, attn, post_downsample in self.downs:
            if exists(pre_downsample):
                x = pre_downsample(x)
            x = init_block(x, t, c)
            for resnet_block in resnet_blocks:
                x = resnet_block(x, t, c)
                hiddens.append(x)
            x = attn(x)
            hiddens.append(x)
            if exists(post_downsample):
                x = post_downsample(x)
        x = self.mid_block1(x, t, mid_c)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x, t, mid_c)

        def connect_skip(fmap):
            return flow.cat((fmap, hiddens.pop() * self.skip_connect_scale), dim=1)

        for init_block, resnet_blocks, attn, upsample in self.ups:
            x = connect_skip(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = connect_skip(x)
                x = resnet_block(x, t, c)

            x = attn(x)
            x = upsample(x)
        x = flow.cat((x, r), dim=1)
        x = self.final_resnet_block(x, t)
        return self.to_out(x)


class LowresConditioner(nn.Module):
    def __init__(
        self,
        downsample_first=True,
        blur_sigma=0.6,
        blur_kernel_size=3,
    ):
        super().__init__()
        self.downsample_first = downsample_first
        self.blur_sigma = blur_sigma
        self.blur_kernel_size = blur_kernel_size

    def forward(
        self,
        cond_fmap,
        *,
        target_image_size,
        downsample_image_size=None,
        blur_sigma=None,
        blur_kernel_size=None,
    ):
        if self.training and self.downsample_first and exists(downsample_image_size):
            cond_fmap = resize_image_to(cond_fmap, downsample_image_size)

        if self.training:
            # when training, blur the low resolution conditional image
            blur_sigma = default(blur_sigma, self.blur_sigma)
            blur_kernel_size = default(blur_kernel_size, self.blur_kernel_size)

            # allow for drawing a random sigma between lo and hi float values
            if isinstance(blur_sigma, tuple):
                blur_sigma = tuple(map(float, blur_sigma))
                blur_sigma = random.uniform(*blur_sigma)

            # allow for drawing a random kernel size between lo and hi int values
            if isinstance(blur_kernel_size, tuple):
                blur_kernel_size = tuple(map(int, blur_kernel_size))
                kernel_size_lo, kernel_size_hi = blur_kernel_size
                blur_kernel_size = random.randrange(kernel_size_lo, kernel_size_hi + 1)

            cond_fmap = gaussian_blur2d(
                cond_fmap, cast_tuple(blur_kernel_size, 2), cast_tuple(blur_sigma, 2)
            )

        cond_fmap = resize_image_to(cond_fmap, target_image_size)

        return cond_fmap


class Decoder(nn.Module):
    def __init__(
        self,
        unet,
        *,
        clip=None,
        image_size=None,
        channels=3,
        vae=tuple(),
        timesteps=1000,
        image_cond_drop_prob=0.1,
        text_cond_drop_prob=0.5,
        loss_type="l2",
        beta_schedule=None,
        predict_x_start=False,
        predict_x_start_for_latent_diffusion=False,
        image_sizes=None,  # for cascading ddpm, image size at each stage
        random_crop_sizes=None,
        lowres_downsample_first=True,
        blur_sigma=0.6,  # cascading ddpm - blur sigma
        blur_kernel_size=3,  # cascading ddpm - blur kernel size
        clip_denoised=True,
        clip_x_start=True,
        clip_adapter_overrides=dict(),
        learned_variance=True,
        learned_variance_constrain_frac=False,
        vb_loss_weight=0.001,
        unconditional=False,  # set to True for generating images without conditioning
        auto_normalize_img=True,
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.9,
        p2_loss_weight_gamma=0.0,
        p2_loss_weight_k=1,
    ):
        super().__init__()

        # clip

        self.clip = None
        if exists(clip):
            assert not unconditional, "clip must not be given if doing unconditional image training"
            assert (
                channels == clip.image_channels
            ), f"channels of image ({channels}) should be equal to the"
            " channels that CLIP accepts ({clip.image_channels})"

            freeze_model_and_make_eval_(clip)
            self.clip = clip

        # determine image size, with image_size and image_sizes taking precedence

        if exists(image_size) or exists(image_sizes):
            assert exists(image_size) ^ exists(
                image_sizes
            ), "only one of image_size or image_sizes must be given"
            image_size = default(image_size, lambda: image_sizes[-1])
        elif exists(clip):
            image_size = clip.image_size
        else:
            raise ("either image_size, image_sizes, or clip must be given to decoder")

        # channels

        self.channels = channels

        # verify conditioning method

        unets = cast_tuple(unet)
        num_unets = len(unets)

        self.unconditional = unconditional

        # automatically take care of ensuring that first unet is unconditional while the rest
        # of the unets are conditioned on the low resolution image produced by previous unet

        vaes = pad_tuple_to_length(
            cast_tuple(vae), len(unets), fillvalue=NullVQGanVAE(channels=self.channels)
        )

        # whether to use learned variance, defaults to True for the first unet in the cascade

        learned_variance = pad_tuple_to_length(
            cast_tuple(learned_variance), len(unets), fillvalue=False
        )
        self.learned_variance = learned_variance
        # whether to constrain the output of the network (the interpolation fraction) from 0 to 1
        self.learned_variance_constrain_frac = learned_variance_constrain_frac
        self.vb_loss_weight = vb_loss_weight

        # construct unets and vaes

        self.unets = nn.ModuleList([])
        self.vaes = nn.ModuleList([])

        for ind, (one_unet, one_vae, one_unet_learned_var) in enumerate(
            zip(unets, vaes, learned_variance)
        ):
            assert isinstance(one_unet, Unet)
            assert isinstance(one_vae, (VQGanVAE, NullVQGanVAE))

            is_first = ind == 0
            latent_dim = one_vae.encoded_dim if exists(one_vae) else None

            unet_channels = default(latent_dim, self.channels)
            unet_channels_out = unet_channels * (1 if not one_unet_learned_var else 2)

            one_unet = one_unet.cast_model_parameters(
                lowres_cond=not is_first,
                cond_on_image_embeds=not unconditional and is_first,
                cond_on_text_encodings=not unconditional and one_unet.cond_on_text_encodings,
                channels=unet_channels,
                channels_out=unet_channels_out,
            )

            self.unets.append(one_unet)
            self.vaes.append(one_vae.copy_for_eval())

        # determine from unets whether conditioning on text encoding is needed

        self.condition_on_text_encodings = any([unet.cond_on_text_encodings for unet in self.unets])

        # create noise schedulers per unet

        if not exists(beta_schedule):
            beta_schedule = (
                "cosine",
                *(("cosine",) * max(num_unets - 2, 0)),
                *(("linear",) * int(num_unets > 1)),
            )

        beta_schedule = cast_tuple(beta_schedule, num_unets)
        p2_loss_weight_gamma = cast_tuple(p2_loss_weight_gamma, num_unets)

        self.noise_schedulers = nn.ModuleList([])

        for unet_beta_schedule, unet_p2_loss_weight_gamma in zip(
            beta_schedule, p2_loss_weight_gamma
        ):
            noise_scheduler = NoiseScheduler(
                beta_schedule=unet_beta_schedule,
                timesteps=timesteps,
                loss_type=loss_type,
                p2_loss_weight_gamma=unet_p2_loss_weight_gamma,
                p2_loss_weight_k=p2_loss_weight_k,
            )

            self.noise_schedulers.append(noise_scheduler)

        # unet image sizes

        image_sizes = default(image_sizes, (image_size,))
        image_sizes = tuple(sorted(set(image_sizes)))

        assert len(self.unets) == len(
            image_sizes
        ), "you did not supply the correct number of u-nets "
        f"({len(self.unets)}) for resolutions {image_sizes}"
        self.image_sizes = image_sizes
        self.sample_channels = cast_tuple(self.channels, len(image_sizes))

        # random crop sizes (for super-resoluting unets at the end of cascade?)

        self.random_crop_sizes = cast_tuple(random_crop_sizes, len(image_sizes))

        # predict x0 config

        self.predict_x_start = (
            cast_tuple(predict_x_start, len(unets))
            if not predict_x_start_for_latent_diffusion
            else tuple(map(lambda t: isinstance(t, VQGanVAE), self.vaes))
        )

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (
            False,
            *((True,) * (len(self.unets) - 1)),
        ), "the first unet must be unconditioned (by low resolution image), "
        "and the rest of the unets must have `lowres_cond` set to True"

        self.to_lowres_cond = LowresConditioner(
            downsample_first=lowres_downsample_first,
            blur_sigma=blur_sigma,
            blur_kernel_size=blur_kernel_size,
        )

        # classifier free guidance

        self.image_cond_drop_prob = image_cond_drop_prob
        self.text_cond_drop_prob = text_cond_drop_prob
        self.can_classifier_guidance = image_cond_drop_prob > 0.0 or text_cond_drop_prob > 0.0

        # whether to clip when sampling

        self.clip_denoised = clip_denoised
        self.clip_x_start = clip_x_start

        # dynamic thresholding settings, if clipping denoised during sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity

        # device tracker

        self.register_buffer("_dummy", flow.Tensor([True]), persistent=False)

    def get_unet(self, unet_number):
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1
        return self.unets[index]

    @contextmanager
    def one_unet_in_gpu(self, unet_number=None, unet=None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.get_unet(unet_number)

        self.cuda()
        devices = [module_device(unet) for unet in self.unets]
        self.unets.cpu()
        unet.cuda()
        yield
        for unet, device in zip(self.unets, devices):
            unet.to(device)

    def p_mean_variance(
        self,
        unet,
        x,
        t,
        image_embed,
        noise_scheduler,
        text_encodings=None,
        text_mask=None,
        lowres_cond_img=None,
        clip_denoised=True,
        predict_x_start=False,
        learned_variance=False,
        cond_scale=1.0,
        model_output=None,
    ):
        assert not (
            cond_scale != 1.0 and not self.can_classifier_guidance
        ), "the decoder was not trained with conditional dropout, "
        "and thus one cannot use classifier free guidance (cond_scale anything other than 1)"

        pred = default(
            model_output,
            lambda: unet.forward_with_cond_scale(
                x,
                t,
                image_embed=image_embed,
                text_encodings=text_encodings,
                text_mask=text_mask,
                cond_scale=cond_scale,
                lowres_cond_img=lowres_cond_img,
            ),
        )

        if learned_variance:
            pred, var_interp_frac_unnormalized = pred.chunk(2, dim=1)
        if predict_x_start:
            x_recon = pred
        else:
            x_recon = noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        if clip_denoised:
            # s is the threshold amount
            # static thresholding would just be s = 1
            s = 1.0
            if self.use_dynamic_thres:
                s = flow.quantile(
                    rearrange(x_recon, "b ... -> b (...)").abs(),
                    self.dynamic_thres_percentile,
                    dim=-1,
                )

                s.clamp_(min=1.0)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = noise_scheduler.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )

        if learned_variance:
            # if learned variance, posterio variance and posterior log variance are
            # predicted by the network by an interpolation of the max and min log beta values
            # eq 15 - https://arxiv.org/abs/2102.09672
            min_log = extract(noise_scheduler.posterior_log_variance_clipped, t, x.shape)
            max_log = extract(flow.log(noise_scheduler.betas), t, x.shape)
            var_interp_frac = unnormalize_zero_to_one(var_interp_frac_unnormalized)

            if self.learned_variance_constrain_frac:
                var_interp_frac = var_interp_frac.sigmoid()

            posterior_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log
            posterior_variance = posterior_log_variance.exp()

        return model_mean, posterior_variance, posterior_log_variance

    @flow.no_grad()
    def p_sample(
        self,
        unet,
        x,
        t,
        image_embed,
        noise_scheduler,
        text_encodings=None,
        text_mask=None,
        cond_scale=1.0,
        lowres_cond_img=None,
        predict_x_start=False,
        learned_variance=False,
        clip_denoised=True,
    ):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            unet,
            x=x,
            t=t,
            image_embed=image_embed,
            text_encodings=text_encodings,
            text_mask=text_mask,
            cond_scale=cond_scale,
            lowres_cond_img=lowres_cond_img,
            clip_denoised=clip_denoised,
            predict_x_start=predict_x_start,
            noise_scheduler=noise_scheduler,
            learned_variance=learned_variance,
        )
        noise = flow.randn(*x.shape, placement=get_default_placement(), sbp=get_default_sbp())
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @flow.no_grad()
    def p_sample_loop(
        self,
        unet,
        shape,
        image_embed,
        noise_scheduler,
        predict_x_start=False,
        learned_variance=False,
        clip_denoised=True,
        lowres_cond_img=None,
        text_encodings=None,
        text_mask=None,
        cond_scale=1,
        is_latent_diffusion=False,
    ):

        b = shape[0]
        img = flow.randn(*shape, placement=get_default_placement(), sbp=get_default_sbp())

        if not is_latent_diffusion:
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        for i in tqdm(
            reversed(range(0, noise_scheduler.num_timesteps)),
            desc="sampling loop time step",
            total=noise_scheduler.num_timesteps,
        ):
            img = self.p_sample(
                unet,
                img,
                flow.full(
                    (b,),
                    i,
                    placement=get_default_placement(),
                    sbp=get_default_sbp(),
                    dtype=flow.long,
                ),
                image_embed=image_embed,
                text_encodings=text_encodings,
                text_mask=text_mask,
                cond_scale=cond_scale,
                lowres_cond_img=lowres_cond_img,
                predict_x_start=predict_x_start,
                noise_scheduler=noise_scheduler,
                learned_variance=learned_variance,
                clip_denoised=clip_denoised,
            )

        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img

    def p_losses(
        self,
        unet,
        x_start,
        times,
        *,
        image_embed,
        noise_scheduler,
        lowres_cond_img=None,
        text_encodings=None,
        text_mask=None,
        predict_x_start=False,
        noise=None,
        learned_variance=False,
        clip_denoised=False,
        is_latent_diffusion=False,
    ):
        noise = default(
            noise,
            lambda: flow.randn(
                *x_start.shape, placement=get_default_placement(), sbp=get_default_sbp()
            ),
        )

        # normalize to [-1, 1]

        if not is_latent_diffusion:
            x_start = self.normalize_img(x_start)
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # get x_t

        x_noisy = noise_scheduler.q_sample(x_start=x_start, t=times, noise=noise)

        model_output = unet(
            x_noisy,
            times,
            image_embed=image_embed,
            text_encodings=text_encodings,
            text_mask=text_mask,
            lowres_cond_img=lowres_cond_img,
            image_cond_drop_prob=self.image_cond_drop_prob,
            text_cond_drop_prob=self.text_cond_drop_prob,
        )

        if learned_variance:
            pred, _ = model_output.chunk(2, dim=1)
        else:
            pred = model_output

        target = noise if not predict_x_start else x_start

        loss = noise_scheduler.loss_fn(pred, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")

        loss = noise_scheduler.p2_reweigh_loss(loss, times)

        loss = loss.mean()

        if not learned_variance:
            # return simple loss if not using learned variance
            return loss

        # most of the code below is transcribed from
        # https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
        # the Improved DDPM paper then further modified it so that the mean is detached
        # (shown a couple lines before), and weighted to be smaller than the l1 or l2 "simple" loss
        # it is questionable whether this is really needed, looking at some of the figures in the
        # paper, but may as well stay faithful to their implementation

        # if learning the variance, also include the extra weight kl loss

        true_mean, _, true_log_variance_clipped = noise_scheduler.q_posterior(
            x_start=x_start, x_t=x_noisy, t=times
        )
        model_mean, _, model_log_variance = self.p_mean_variance(
            unet,
            x=x_noisy,
            t=times,
            image_embed=image_embed,
            noise_scheduler=noise_scheduler,
            clip_denoised=clip_denoised,
            learned_variance=True,
            model_output=model_output,
        )

        # kl loss with detached model predicted mean, for stability reasons as in paper

        detached_model_mean = model_mean.detach()

        kl = normal_kl(
            true_mean, true_log_variance_clipped, detached_model_mean, model_log_variance
        )
        kl = meanflat(kl) * NAT

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=detached_model_mean, log_scales=0.5 * model_log_variance
        )
        decoder_nll = meanflat(decoder_nll) * NAT

        # at the first timestep return the decoder NLL,
        # otherwise KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))

        vb_losses = flow.where(times == 0, decoder_nll, kl)

        # weight the vb loss smaller, for stability, as in the paper (recommended 0.001)

        vb_loss = vb_losses.mean() * self.vb_loss_weight

        return loss + vb_loss

    @flow.no_grad()
    @eval_decorator
    def sample(
        self,
        image_embed=None,
        text=None,
        text_mask=None,
        text_encodings=None,
        batch_size=1,
        cond_scale=1.0,
        stop_at_unet_number=None,
        distributed=False,
    ):
        assert self.unconditional or exists(
            image_embed
        ), "image embed must be present on sampling from decoder unless if trained unconditionally"

        if not self.unconditional:
            batch_size = image_embed.shape[0]

        if exists(text) and not exists(text_encodings) and not self.unconditional:
            assert exists(self.clip)
            _, text_encodings, text_mask = self.clip.embed_text(text)

        assert not (
            self.condition_on_text_encodings and not exists(text_encodings)
        ), "text or text encodings must be passed into decoder if specified"
        assert not (
            not self.condition_on_text_encodings and exists(text_encodings)
        ), "decoder specified not to be conditioned on text, yet it is presented"

        img = None

        for (
            unet_number,
            unet,
            vae,
            channel,
            image_size,
            predict_x_start,
            learned_variance,
            noise_scheduler,
        ) in tqdm(
            zip(
                range(1, len(self.unets) + 1),
                self.unets,
                self.vaes,
                self.sample_channels,
                self.image_sizes,
                self.predict_x_start,
                self.learned_variance,
                self.noise_schedulers,
            )
        ):

            context = null_context()

            with context:
                lowres_cond_img = None
                shape = (batch_size, channel, image_size, image_size)

                if unet.lowres_cond:
                    lowres_cond_img = self.to_lowres_cond(img, target_image_size=image_size)

                is_latent_diffusion = isinstance(vae, VQGanVAE)
                image_size = vae.get_encoded_fmap_size(image_size)
                shape = (batch_size, vae.encoded_dim, image_size, image_size)

                lowres_cond_img = maybe(vae.encode)(lowres_cond_img)

                img = self.p_sample_loop(
                    unet,
                    shape,
                    image_embed=image_embed,
                    text_encodings=text_encodings,
                    text_mask=text_mask,
                    cond_scale=cond_scale,
                    predict_x_start=predict_x_start,
                    learned_variance=learned_variance,
                    clip_denoised=not is_latent_diffusion,
                    lowres_cond_img=lowres_cond_img,
                    is_latent_diffusion=is_latent_diffusion,
                    noise_scheduler=noise_scheduler,
                )

                img = vae.decode(img)

            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        return img

    def forward(
        self,
        image,
        text=None,
        image_embed=None,
        text_encodings=None,
        text_mask=None,
        unet_number=None,
        return_lowres_cond_image=False,
    ):
        assert not (
            len(self.unets) > 1 and not exists(unet_number)
        ), f"you must specify which unet you want trained, from a range of 1 to {len(self.unets)},"
        " if you are training cascading DDPM (multiple unets)"
        unet_number = default(unet_number, 1)
        unet_index = unet_number - 1

        unet = self.get_unet(unet_number)

        vae = self.vaes[unet_index]
        noise_scheduler = self.noise_schedulers[unet_index]
        target_image_size = self.image_sizes[unet_index]
        predict_x_start = self.predict_x_start[unet_index]
        random_crop_size = self.random_crop_sizes[unet_index]
        learned_variance = self.learned_variance[unet_index]
        b, _, h, w, _, = (
            *image.shape,
            image.device,
        )

        check_shape(image, "b c h w", c=self.channels)
        assert h >= target_image_size and w >= target_image_size

        times = flow.randint(
            0,
            noise_scheduler.num_timesteps,
            (b,),
            placement=get_default_placement(),
            sbp=get_default_sbp(),
            dtype=flow.long,
        )

        if not exists(image_embed) and not self.unconditional:
            assert exists(self.clip), "if you want to derive CLIP image embeddings automatically, "
            "you must supply `clip` to the decoder on init"
            image_embed, _ = self.clip.embed_image(image)

        if exists(text) and not exists(text_encodings) and not self.unconditional:
            assert exists(
                self.clip
            ), "if you are passing in raw text, you need to supply `clip` to the decoder"
            _, text_encodings, text_mask = self.clip.embed_text(text)

        assert not (
            self.condition_on_text_encodings and not exists(text_encodings)
        ), "text or text encodings must be passed into decoder if specified"
        assert not (
            not self.condition_on_text_encodings and exists(text_encodings)
        ), "decoder specified not to be conditioned on text, yet it is presented"

        lowres_cond_img = (
            self.to_lowres_cond(
                image,
                target_image_size=target_image_size,
                downsample_image_size=self.image_sizes[unet_index - 1],
            )
            if unet_number > 1
            else None
        )
        image = resize_image_to(image, target_image_size)

        if exists(random_crop_size):
            aug = K.RandomCrop((random_crop_size, random_crop_size), p=1.0)
            # make sure low res conditioner and image both get augmented the same way
            image = aug(image)
            lowres_cond_img = aug(lowres_cond_img, params=aug._params)

        is_latent_diffusion = not isinstance(vae, NullVQGanVAE)

        vae.eval()
        with flow.no_grad():
            image = vae.encode(image)
            lowres_cond_img = maybe(vae.encode)(lowres_cond_img)

        losses = self.p_losses(
            unet,
            image,
            times,
            image_embed=image_embed,
            text_encodings=text_encodings,
            text_mask=text_mask,
            lowres_cond_img=lowres_cond_img,
            predict_x_start=predict_x_start,
            learned_variance=learned_variance,
            is_latent_diffusion=is_latent_diffusion,
            noise_scheduler=noise_scheduler,
        )

        if not return_lowres_cond_image:
            return losses

        return losses, lowres_cond_img


# main class


class DALLE2(nn.Module):
    def __init__(self, *, prior, decoder, prior_num_samples=2, **kwargs):
        super().__init__()
        # assert isinstance(prior, DiffusionPrior)
        # assert isinstance(decoder, Decoder)
        self.prior = prior
        self.decoder = decoder
        self.tokenizer = SimpleTokenizer()

        self.prior_num_samples = prior_num_samples
        self.decoder_need_text_cond = self.decoder.condition_on_text_encodings

        self.to_pil = T.ToPILImage()

    @flow.no_grad()
    @eval_decorator
    def forward(self, text, cond_scale=1.0, prior_cond_scale=1.0, return_pil_images=False):
        device = module_device(self)
        one_text = isinstance(text, str) or (not is_list_str(text) and text.shape[0] == 1)

        if isinstance(text, str) or is_list_str(text):
            text = [text] if not isinstance(text, (list, tuple)) else text
            text = self.tokenizer.tokenize(text).to(device)

        image_embed = self.prior.sample(
            text, num_samples_per_batch=self.prior_num_samples, cond_scale=prior_cond_scale
        )

        text_cond = text if self.decoder_need_text_cond else None
        images = self.decoder.sample(image_embed, text=text_cond, cond_scale=cond_scale)

        if return_pil_images:
            images = list(map(self.to_pil, images.unbind(dim=0)))

        if one_text:
            return first(images)

        return images
