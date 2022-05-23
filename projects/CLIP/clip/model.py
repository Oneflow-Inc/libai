# --------------------------------------------------------
# Borrow code from:
# https://github.com/openai/CLIP/tree/main/clip/model.py
# --------------------------------------------------------

from collections import OrderedDict
from typing import Dict, Tuple, Union

import numpy as np
import oneflow as flow
import torch
from oneflow import nn

from libai.layers import Embedding, LayerNorm, Linear, MultiheadAttention, TransformerLayer
from libai.models import VisionTransformer as ViT
from libai.utils import distributed as dist
from libai.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

from .ops import multi_head_attention_forward


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed
        # after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool,
            # and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: flow.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            flow.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = flow.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=flow.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to flowvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1,
      with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is
      prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.to(dtype=self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: flow.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.attn_mask = attn_mask
        self.resblocks = nn.ModuleList(
            [TransformerLayer(width, 4 * width, heads, layer_idx=i) for i in range(layers)]
        )

    def forward(self, x: flow.Tensor):
        for layer in self.resblocks:
            x = layer(x, self.attn_mask)
        return x


class VisionTransformer(ViT):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0,
        num_classes=1000,
        loss_func=None,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            num_classes,
            loss_func,
        )

        self.ln_pre = LayerNorm(embed_dim, layer_idx=0)
        self.head = Linear(embed_dim, num_classes, bias=False, layer_idx=-1)

    def forward_features(self, x):
        # patch embedding
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        cls_token = cls_token.to_global(sbp=x.sbp, placement=cls_token.placement)
        x = flow.cat((cls_token, x), dim=1)

        # position embedding
        pos_embed = self.pos_embed.expand(x.shape[0], -1, -1)
        pos_embed = pos_embed.to_global(sbp=x.sbp, placement=pos_embed.placement)
        x = self.pos_drop(x + pos_embed)

        # layernorm_pre
        x = self.ln_pre(x)

        # transformer block
        x = self.blocks(x)
        return x


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            ).to_global(sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0))
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                img_size=image_resolution,
                patch_size=vision_patch_size,
                embed_dim=vision_width,
                depth=vision_layers,
                num_heads=vision_heads,
                num_classes=embed_dim,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            flow.empty(
                self.context_length,
                transformer_width,
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(0),
            )
        )
        self.ln_final = LayerNorm((transformer_width,), layer_idx=-1)

        self.text_projection = nn.Parameter(
            flow.empty(
                transformer_width,
                embed_dim,
                sbp=flow.sbp.broadcast,
                placement=dist.get_layer_placement(0),
            )
        )
        self.logit_scale = nn.Parameter(
            flow.ones([], sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0))
            * np.log(1 / 0.07)
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        if hasattr(self.visual, "patch_embed"):
            nn.init.zeros_(self.visual.patch_embed.proj.bias)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4,
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attention.query_key_value.weight, std=attn_std)
            nn.init.normal_(block.attention.dense.weight, std=proj_std)
            nn.init.normal_(block.mlp.dense_h_to_4h.weight, std=fc_std)
            nn.init.normal_(block.mlp.dense_4h_to_h.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = flow.ones(
            self.context_length,
            self.context_length,
            sbp=flow.sbp.broadcast,
            placement=dist.get_layer_placement(0),
        )
        mask = flow.tril(mask)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image)["prediction_scores"]

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[flow.arange(x.shape[0], sbp=x.sbp, placement=x.placement), text.argmax(dim=-1)]
            @ self.text_projection
        )

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype=flow.float16)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype=flow.float16)

        if isinstance(l, MultiheadAttention):
            for attr in ["query_key_value", "dense"]:
                layer = getattr(l, attr)
                weight = getattr(layer, "weight")
                if weight is not None:
                    weight.data = weight.data.to(dtype=flow.float16)
                bias = getattr(layer, "bias")
                if bias is not None:
                    bias.data = bias.data.to(dtype=flow.float16)

        if hasattr(l, "text_projection"):
            attr = getattr(l, "text_projection")
            if attr is not None:
                attr.data = attr.data.to(dtype=flow.float16)

        if hasattr(l, "proj"):
            attr = getattr(l, "proj")
            if attr is not None:
                attr.weight.data = attr.weight.data.to(dtype=flow.float16)

    model.apply(_convert_weights_to_fp16)


def load_tensor(tensor_lhs: flow.Tensor, tensor_rhs: torch.Tensor):
    tensor_rhs = flow.Tensor(
        tensor_rhs.cpu().numpy(),
        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        placement=flow.env.all_device_placement("cuda"),
    ).to_global(sbp=tensor_lhs.sbp, placement=tensor_lhs.placement)
    tensor_lhs.data.copy_(tensor_rhs.data)


def load_weights(model: nn.Module, state_dict: Dict):
    model_state_dict = model.state_dict()
    incorrect_shapes = []
    for k in list(state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(state_dict[k].shape)
            if shape_model != shape_checkpoint:
                incorrect_shapes.append((k, shape_checkpoint, shape_model))
                state_dict.pop(k)

    unexpected_keys = []
    for key, value in state_dict.items():
        if key not in model_state_dict:
            unexpected_keys.append(key)
            # skip this key
            continue
        model_state_dict.pop(key)
        load_tensor(model.state_dict()[key], value)

    missing_keys = list(model_state_dict.keys())
    for k, shape_checkpoint, shape_model in incorrect_shapes:
        print(
            "Skip loading parameter '{}' to the model due to incompatible "
            "shapes: {} in the checkpoint but {} in the "
            "model! You might want to double check if this is expected.".format(
                k, shape_checkpoint, shape_model
            )
        )
    if missing_keys:
        print(get_missing_parameters_message(missing_keys))
    if unexpected_keys:
        print(get_unexpected_parameters_message(unexpected_keys))


def convert_qkv_weight(qkv_weight, num_heads):
    qkv_weight = qkv_weight.view([3, num_heads, 64, num_heads * 64])
    qkv_weight = (
        qkv_weight.permute(1, 0, 2, 3).contiguous().view(3 * num_heads * 64, num_heads * 64)
    )
    return qkv_weight


def convert_qkv_bias(qkv_bias, num_heads):
    qkv_bias = qkv_bias.view(3, num_heads, 64)
    qkv_bias = qkv_bias.permute(1, 0, 2).contiguous().view(-1)
    return qkv_bias


def change_vit_state_dict(state_dict, visual_num_heads, text_num_heads):
    new_state_dict = {}
    for key, value in state_dict.items():
        # change prefix
        if "visual.transformer.resblocks" in key:
            key = key.replace("visual.transformer.resblocks", "visual.blocks")
        # change "ln_1" to "input_layernorm"
        if "ln_1" in key:
            key = key.replace("ln_1", "input_layernorm")
        # change "ln_2" to "post_attention_layernorm"
        if "ln_2" in key:
            key = key.replace("ln_2", "post_attention_layernorm")
        # change "attn.out_proj" to "attention.dense"
        if "attn.out_proj" in key:
            key = key.replace("attn.out_proj", "attention.dense")
        # change "attn" to "attention.query_key_value"
        if "attn.in_proj_weight" in key:
            key = key.replace("attn.in_proj_weight", "attention.query_key_value.weight")
            if "visual" not in key:
                value = convert_qkv_weight(value, text_num_heads)
            else:
                value = convert_qkv_weight(value, visual_num_heads)
        if "attn.in_proj_bias" in key:
            key = key.replace("attn.in_proj_bias", "attention.query_key_value.bias")
            if "visual" not in key:
                value = convert_qkv_bias(value, text_num_heads)
            else:
                value = convert_qkv_bias(value, visual_num_heads)
        # change "mlp.c_fc" to "mlp.dense_h_to_4h"
        if "mlp.c_fc" in key:
            key = key.replace("mlp.c_fc", "mlp.dense_h_to_4h")
        # change "mlp.c_proj" to "mlp.dense_4h_to_h"
        if "mlp.c_proj" in key:
            key = key.replace("mlp.c_proj", "mlp.dense_4h_to_h")

        # change "class_embedding" to "cls_token"
        if "class_embedding" in key:
            key = key.replace("class_embedding", "cls_token")
            value = value.unsqueeze(0).unsqueeze(0)
        # change "pos_embed" to "positional_embedding"
        if "visual.positional_embedding" == key:
            key = "visual.pos_embed"
            value = value.unsqueeze(0)
        # change patch_embedding
        if key == "visual.conv1.weight":
            key = "visual.patch_embed.proj.weight"
        # change "ln_post"
        if "ln_post" in key:
            key = key.replace("ln_post", "norm")
        # change "proj"
        if "visual.proj" == key:
            key = "visual.head.weight"
            value = value.transpose(0, 1)

        new_state_dict[key] = value

    return new_state_dict


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks"))
    )

    if vit:
        state_dict = change_vit_state_dict(state_dict, vision_width // 64, transformer_heads)

    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    load_weights(model, state_dict)
    return model.eval()
