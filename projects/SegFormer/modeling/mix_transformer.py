import math

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
from flowvision.layers import trunc_normal_
from flowvision.models import to_2tuple

from libai.config.config import configurable
from libai.layers import MLP, Linear, LayerNorm, DropPath
from libai.utils import distributed as dist
from projects.SegFormer.modeling.head import DecodeHead


class Mlp(MLP):
    def __init__(self, hidden_size, ffn_hidden_size, output_dropout_prob=0, init_method=nn.init.xavier_normal_, output_layer_init_method=None, bias_gelu_fusion=False, bias_dropout_fusion=False, *, layer_idx=0):
        super(Mlp, self).__init__(hidden_size, ffn_hidden_size, output_dropout_prob, init_method, output_layer_init_method, bias_gelu_fusion, bias_dropout_fusion, layer_idx=layer_idx)
        self.dwconv = DWConv(ffn_hidden_size)
        
    def forward(self, hidden_states, H, W):
        intermediate = self.dense_h_to_4h(hidden_states)
        if self.bias_gelu_fusion:
            intermediate, bias = intermediate
            intermediate = flow._C.fused_bias_add_gelu(
                intermediate, bias, axis=intermediate.ndim - 1
            )
        else:
            intermediate = self.activation_func(intermediate)

        intermediate = self.dwconv(intermediate, H, W)
        output = self.dense_4h_to_h(intermediate)
        
        if self.bias_dropout_fusion:
            output, bias = output
            output = flow._C.fused_bias_add_dropout(
                output, bias, p=self.output_dropout_prob, axis=output.ndim - 1
            )
        else:
            output = self.dropout(output)
        return output


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, layer_idx=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2)).to_global(
                                placement=dist.get_layer_placement(layer_idx),
                                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                              )
        self.norm = LayerNorm(embed_dim, layer_idx=layer_idx)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W
    
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, layer_idx=0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = Linear(dim, dim, bias=qkv_bias, layer_idx=layer_idx)
        self.kv = Linear(dim, dim * 2, bias=qkv_bias, layer_idx=layer_idx)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim, layer_idx=layer_idx)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio).to_global(
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
            self.norm = LayerNorm(dim, layer_idx=layer_idx)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm, sr_ratio=1, layer_idx=0):
        super().__init__()
        self.norm1 = norm_layer(dim, layer_idx=layer_idx)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, layer_idx=layer_idx)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, layer_idx=layer_idx)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim,
                       ffn_hidden_size=mlp_hidden_dim,
                       output_dropout_prob=drop,
                       bias_gelu_fusion=True,
                       bias_dropout_fusion=True,
                       layer_idx=layer_idx)


    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class MixVisionTransformer(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    @configurable
    def __init__(self, img_size=224, patch_sizes=[7, 3, 3, 3], in_chans=3, num_classes=19, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=LayerNorm, loss_func=None,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], decoder_in_channels=[32, 64, 160, 256], decoder_embedding_dim=256, decoder_dropout_prob=0.1, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        strides = [4, 2, 2, 2]
        patch_embeds = []
        for i in range(len(depths)):
            patch_embeds.append(
                OverlapPatchEmbed(patch_size=patch_sizes[i], stride=strides[i], in_chans=in_chans if i == 0 else embed_dims[i-1],
                                  embed_dim=embed_dims[i], layer_idx=i if i == 0 else sum(depths[:i]))
            )
        
        self.patch_embeds = nn.ModuleList(patch_embeds)
        
        # transformer encoder
        dpr = [x.item() for x in flow.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        blocks = []
        cur = 0
        for i in range(len(depths)):
            layers = []
            if i != 0:
                cur += depths[i - 1]
            for j in range(depths[i]):
                layers.append(
                    Block(
                        dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                        sr_ratio=sr_ratios[i], layer_idx=cur + j)
                )
            blocks.append(nn.ModuleList(layers))
            
        self.blocks = nn.ModuleList(blocks)
            
        # Layer norms
        self.layer_norms = nn.ModuleList(
            [norm_layer(embed_dims[i], layer_idx=sum(depths[:i+1]) - 1) for i in range(len(depths))]
        )
        
        # classification head
        self.head = DecodeHead( in_channels=decoder_in_channels,
                                in_index=[0, 1, 2, 3],
                                feature_strides=[4, 8, 16, 32],
                                dropout_ratio=decoder_dropout_prob,
                                embedding_dim=decoder_embedding_dim,
                                num_classes=num_classes,
                                align_corners=False,
                                layer_idx=-1) if num_classes > 0 else nn.Identity()
        
        # Loss func
        self.loss_func = nn.CrossEntropyLoss(ignore_index=ignore_index) if loss_func is None else loss_func

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        
    @classmethod
    def from_config(cls, cfg):
        return {
            'img_size': cfg.img_size,
            'patch_sizes': cfg.patch_sizes,
            'in_chans': cfg.in_chans,
            'num_classes': cfg.num_classes,
            'embed_dims': cfg.embed_dims,
            'num_heads': cfg.num_heads,
            'mlp_ratios': cfg.mlp_ratios,
            'qkv_bias': cfg.qkv_bias,
            'qk_scale': cfg.qk_scale,
            'drop_rate': cfg.drop_rate,
            'attn_drop_rate': cfg.attn_drop_rate,
            'drop_path_rate': cfg.drop_path_rate,
            'depths': cfg.depths,
            'sr_ratios': cfg.sr_ratios,
            'loss_func': cfg.loss_func,
            'decoder_in_channels': cfg.decoder_in_channels,
            'decoder_embedding_dim': cfg.decoder_embedding_dim,
            'decoder_dropout_prob': cfg.decoder_dropout_prob
        }
    

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        
        for idx, m in enumerate(zip(self.patch_embeds, self.blocks, self.layer_norms)):
            embedding_layer, block_layer, norm_layer = m
            x, H, W = embedding_layer(x)
            for i, blk in enumerate(block_layer):
                x = blk(x, H, W)
            x = norm_layer(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)    

        return outs

    def forward(self, images, labels=None):
        x = self.forward_features(images)
        x = self.head(x)
        if labels is not None and self.training:
            x = F.interpolate(x, labels.shape[1:], mode='bilinear')
            losses = self.loss_func(x, labels)
            return {"losses": losses}
        else:
            return {"prediction_scores": x}
        
    @staticmethod
    def set_pipeline_stage_id(model):
        ### TODO set_pipeline_stage_id 
        dist_utils = dist.get_dist_util()
        
        # Set pipeline parallelism stage_id
        for module_block in model.modules():
            # module.origin can get the original module
            if isinstance(module_block.origin, Block):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(module_block.layer_idx),
                    dist.get_layer_placement(module_block.layer_idx),
                )
            elif isinstance(module_block.origin, OverlapPatchEmbed):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(module_block.layer_idx),
                    dist.get_layer_placement(module_block.layer_idx),
                )
            elif isinstance(module_block.origin, LayerNorm):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(module_block.layer_idx),
                    dist.get_layer_placement(module_block.layer_idx),
                )
                
        model.head.config.set_stage(dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1))
        model.loss_func.config.set_stage(
            dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
        )
        
        
    @staticmethod
    def set_activation_checkpoint(model):
        ### TODO set_activation_checkpoint 
        pass    



class DWConv(nn.Module):
    def __init__(self, dim=768, layer_idx=0):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim).to_global(
            placement=dist.get_layer_placement(layer_idx),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        )
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
    

if __name__ == '__main__':
    model = MixVisionTransformer()
    
    for param_tensor in model.state_dict():
        print(param_tensor,'\t',model.state_dict()[param_tensor].size())

    import numpy as np
    input = np.random.rand(1, 3, 512, 1024)
    input = flow.tensor(input, dtype=flow.float32, sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]), placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", [0]),)
    output = model(input)
    print(output['prediction_scores'].shape)
