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
from projects.SegFormer.modeling.mix_transformer import MixVisionTransformer, OverlapPatchEmbed, Block


class MixVisionTransformerForSegmentation(MixVisionTransformer):
    """_summary_

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    @configurable
    def __init__(self, img_size=224, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], in_chans=3, num_classes=19, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=LayerNorm, loss_func=None,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], decoder_in_channels=[32, 64, 160, 256], decoder_embedding_dim=256, decoder_dropout_prob=0.1, ignore_index=255):
        super(MixVisionTransformer, self).__init__(img_size, patch_sizes, strides, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, 
                 loss_func, depths, sr_ratios)
        self.num_classes = num_classes
        

        # segmentation head
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
            'strides': cfg.strides,
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
            print('embedding:', x)
            print(x.sum())
            for i, blk in enumerate(block_layer):
                x = blk(x, H, W)
            x = norm_layer(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)    

        return outs

    def forward(self, images, labels=None):
        x = self.forward_features(images)
        print(x[-1].shape)
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

    # import numpy as np
    # input = np.random.rand(1, 3, 512, 1024)
    # input = flow.tensor(input, dtype=flow.float32, sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]), placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", [0]),)
    # output = model(input)
    # print(output['prediction_scores'].shape)
