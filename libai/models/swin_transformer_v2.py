# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
from flowvision.layers import trunc_normal_
from flowvision.models import to_2tuple

from libai.config.config import configurable
from libai.layers import MLP, DropPath, LayerNorm, Linear
from libai.utils import distributed as dist


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.
                                    Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
        fused_bias_add_dropout=False,
        layer_idx=0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.fused_bias_add_dropout = fused_bias_add_dropout
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.p = proj_drop
        self.logit_scale = nn.Parameter(
            flow.log(
                10
                * flow.ones(
                    1,
                    num_heads,
                    1,
                    1,
                    placement=dist.get_layer_placement(layer_idx),
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                )
            ),
            requires_grad=True,
        )

        # NOTE: generate meta network, using mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            Linear(2, 512, bias=True, layer_idx=layer_idx),
            nn.ReLU(inplace=True),
            Linear(512, num_heads, bias=False, layer_idx=layer_idx),
        ).to_global(
            placement=dist.get_layer_placement(layer_idx),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        )

        # NOTE: get relative_coords_table
        relative_coords_h = flow.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=flow.float32
        )
        relative_coords_w = flow.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=flow.float32
        )
        relative_coords_table = (
            flow.stack(flow.meshgrid(*[relative_coords_h, relative_coords_w]))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2

        # NOTE: For any relative coordinate, constrain it to -8~8 (window size)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] = relative_coords_table[:, :, :, 0] / (
                pretrained_window_size[0] - 1
            )
            relative_coords_table[:, :, :, 1] = relative_coords_table[:, :, :, 1] / (
                pretrained_window_size[1] - 1
            )
        else:
            relative_coords_table[:, :, :, 0] = relative_coords_table[:, :, :, 0] / (
                self.window_size[0] - 1
            )
            relative_coords_table[:, :, :, 1] = relative_coords_table[:, :, :, 1] / (
                self.window_size[1] - 1
            )
        relative_coords_table = relative_coords_table * 8
        # NOTE: y=sign(x)*log(|x|+1)
        relative_coords_table = (
            flow.sign(relative_coords_table)
            * flow.log2(flow.abs(relative_coords_table) + 1.0)
            / math.log2(8.0)
        )
        self.register_buffer(
            "relative_coords_table",
            relative_coords_table.to_global(
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            ),
        )

        # NOTE: get pair-wise relative position index for each token inside the window
        coords_h = flow.arange(self.window_size[0])
        coords_w = flow.arange(self.window_size[1])
        coords = flow.stack(flow.meshgrid(*[coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = flow.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] = (
            relative_coords[:, :, 0] + self.window_size[0] - 1
        )  # shift to start from 0
        relative_coords[:, :, 1] = relative_coords[:, :, 1] + self.window_size[1] - 1
        relative_coords[:, :, 0] = relative_coords[:, :, 0] * (2 * self.window_size[1] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer(
            "relative_position_index",
            relative_position_index.to_global(
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            ),
        )

        self.qkv = Linear(dim, dim * 3, bias=False, layer_idx=layer_idx)
        if qkv_bias:
            self.q_bias = nn.Parameter(
                flow.zeros(
                    dim,
                    placement=dist.get_layer_placement(layer_idx),
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                )
            )
            self.v_bias = nn.Parameter(
                flow.zeros(
                    dim,
                    placement=dist.get_layer_placement(layer_idx),
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                )
            )
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim, layer_idx=layer_idx)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = flow.concat(
                [
                    self.q_bias,
                    flow.zeros(
                        self.v_bias.shape,
                        requires_grad=False,
                        placement=dist.get_layer_placement(
                            self.layer_idx, device_type=self.v_bias.placement.type
                        ),
                        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                    ),
                    self.v_bias,
                ],
                dim=0,
            )
        qkv = self.qkv(x) + qkv_bias.unsqueeze(0).unsqueeze(0)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # NOTE: cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)

        # NOTE: a learnable scalar
        logit_scale = flow.clamp(self.logit_scale, min=-1e6, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale

        # NOTE: use relative_coords_table and meta network to generate relative_position_bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(
            -1, self.num_heads
        )
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww

        # NOTE: constrained to a range of -16~16
        relative_position_bias = 16 * flow.sigmoid(relative_position_bias).unsqueeze(0)
        attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if self.fused_bias_add_dropout:
            x = flow._C.matmul(x, self.proj.weight, transpose_a=False, transpose_b=True)
            x = flow._C.fused_bias_add_dropout(x, self.proj.bias, p=self.p, axis=2)
        else:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
        pretrained_window_size=0,
        layer_idx=0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.layer_idx = layer_idx
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim, layer_idx=layer_idx)

        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
            fused_bias_add_dropout=True,
            layer_idx=layer_idx,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim, layer_idx=layer_idx)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            hidden_size=dim,
            ffn_hidden_size=mlp_hidden_dim,
            output_dropout_prob=drop,
            bias_gelu_fusion=True,
            bias_dropout_fusion=True,
            layer_idx=layer_idx,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = flow.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt = cnt + 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = (
                attn_mask.masked_fill(attn_mask != 0, float(-100.0))
                .masked_fill(attn_mask == 0, float(0.0))
                .to_global(
                    placement=dist.get_layer_placement(layer_idx),
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                )
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = flow.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = flow.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        # NOTE: res-post-norm
        x = shortcut + self.drop_path(self.norm1(x))

        # NOTE: res-post-norm
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: libai.layers.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=LayerNorm, layer_idx=0):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Linear(4 * dim, 2 * dim, bias=False, layer_idx=layer_idx)

        # NOTE: swinv2-> 2*dim, swin-> 4*dim
        self.norm = norm_layer(2 * dim, layer_idx=layer_idx)
        self.layer_idx = layer_idx

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = flow.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # NOTE: post-res-norm, a change that swin-v2 compared to swin
        x = self.reduction(x)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
                                                                            Default: None
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=LayerNorm,
        downsample=None,
        pretrained_window_size=0,
        layer_id_offset=0,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.layer_id_offset = layer_id_offset
        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size,
                    layer_idx=layer_id_offset + i,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                norm_layer=norm_layer,
                layer_idx=layer_id_offset + depth - 1,
            )
        else:
            self.downsample = None

    def forward(self, x):
        layer_idx = self.layer_id_offset
        for blk in self.blocks:
            x = x.to_global(
                placement=dist.get_layer_placement(layer_idx, device_type=x.placement.type)
            )
            x = blk(x)
            layer_idx += 1
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, layer_idx=0
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        ).to_global(
            placement=dist.get_layer_placement(layer_idx),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model \
        ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerV2(nn.Module):
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    @configurable
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=LayerNorm,
        ape=False,
        patch_norm=True,
        pretrained_window_sizes=[0, 0, 0, 0],
        loss_func=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            layer_idx=0,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                flow.zeros(
                    1,
                    num_patches,
                    embed_dim,
                    placement=dist.get_layer_placement(0),
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                )
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in flow.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        layer_id_offset = 0
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                pretrained_window_size=pretrained_window_sizes[i_layer],
                layer_id_offset=layer_id_offset,
            )
            layer_id_offset += depths[i_layer]
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features, layer_idx=-1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            Linear(self.num_features, num_classes, layer_idx=-1)
            if num_classes > 0
            else nn.Identity()
        )
        self.loss_func = nn.CrossEntropyLoss() if loss_func is None else loss_func

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @classmethod
    def from_config(cls, cfg):
        return {
            "img_size": cfg.img_size,
            "patch_size": cfg.patch_size,
            "in_chans": cfg.in_chans,
            "num_classes": cfg.num_classes,
            "embed_dim": cfg.embed_dim,
            "depths": cfg.depths,
            "num_heads": cfg.num_heads,
            "window_size": cfg.window_size,
            "mlp_ratio": cfg.mlp_ratio,
            "qkv_bias": cfg.qkv_bias,
            "drop_rate": cfg.drop_rate,
            "drop_path_rate": cfg.drop_path_rate,
            "ape": cfg.ape,
            "patch_norm": cfg.patch_norm,
            "pretrained_window_sizes": cfg.pretrained_window_sizes,
            "loss_func": cfg.loss_func,
        }

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = flow.flatten(x, 1)
        return x

    def forward(self, images, labels=None):
        """

        Args:
            images (flow.Tensor): training samples.
            labels (flow.LongTensor, optional): training targets

        Returns:
            dict:
                A dict containing :code:`loss_value` or :code:`logits`
                depending on training or evaluation mode.
                :code:`{"losses": loss_value}` when training,
                :code:`{"prediction_scores": logits}` when evaluating.
        """
        x = self.forward_features(images)
        x = self.head(x)

        if labels is not None and self.training:
            losses = self.loss_func(x, labels)
            return {"losses": losses}
        else:
            return {"prediction_scores": x}

    @staticmethod
    def set_pipeline_stage_id(model):
        dist_utils = dist.get_dist_util()

        # Set pipeline parallelism stage_id
        if hasattr(model.patch_embed, "config"):
            # Old API in OneFlow 0.8
            model.patch_embed.config.set_stage(
                dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
            )
            model.pos_drop.config.set_stage(
                dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
            )

            for module_block in model.modules():
                if isinstance(module_block.origin, SwinTransformerBlock):
                    module_block.config.set_stage(
                        dist_utils.get_layer_stage_id(module_block.layer_idx),
                        dist.get_layer_placement(module_block.layer_idx),
                    )
                elif isinstance(module_block.origin, PatchMerging):
                    module_block.config.set_stage(
                        dist_utils.get_layer_stage_id(module_block.layer_idx),
                        dist.get_layer_placement(module_block.layer_idx),
                    )

            model.norm.config.set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
            model.head.config.set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
            model.avgpool.config.set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
            model.loss_func.config.set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
        else:
            model.patch_embed.to(flow.nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
            )
            model.pos_drop.to(flow.nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
            )

            for module_block in model.modules():
                if isinstance(module_block.to(nn.Module), SwinTransformerBlock):
                    module_block.to(flow.nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(module_block.layer_idx),
                        dist.get_layer_placement(module_block.layer_idx),
                    )
                elif isinstance(module_block.to(nn.Module), PatchMerging):
                    module_block.to(flow.nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(module_block.layer_idx),
                        dist.get_layer_placement(module_block.layer_idx),
                    )

            model.norm.to(flow.nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
            model.head.to(flow.nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
            model.avgpool.to(flow.nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
            model.loss_func.to(flow.nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )

    @staticmethod
    def set_activation_checkpoint(model):
        for module_block in model.modules():
            if hasattr(module_block, "origin"):
                # Old API in OneFlow 0.8
                if isinstance(module_block.origin, SwinTransformerBlock):
                    module_block.config.activation_checkpointing = True
            else:
                if isinstance(module_block.to(nn.Module), SwinTransformerBlock):
                    module_block.to(flow.nn.graph.GraphModule).activation_checkpointing = True
