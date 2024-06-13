import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
from flowvision.layers import trunc_normal_
from flowvision.models import to_2tuple

from libai.config.config import configurable
from libai.layers import LayerNorm
from libai.utils import distributed as dist
from projects.SegFormer.modeling.head import DecodeHead
from projects.SegFormer.modeling.segformer_model import SegformerModel, Block, OverlapPatchEmbed
from projects.SegFormer.model_utils.load_pretrained_imagenet1k import SegFLoaderImageNet1kPretrain

class SegformerSegmentationLoadImageNetPretrain(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.pretrained_model_path is not None:
            loader = SegFLoaderImageNet1kPretrain(SegformerModel, cfg, cfg.pretrained_model_path)
            self.segformer = loader.load()
        else:
            self.segformer = SegformerModel(cfg)
        self.num_classes = cfg.num_classes
        self.head = DecodeHead(in_channels=cfg.decoder_in_channels,
                                in_index=[0, 1, 2, 3],
                                feature_strides=[4, 8, 16, 32],
                                dropout_ratio=cfg.decoder_dropout_prob,
                                embedding_dim=cfg.decoder_embedding_dim,
                                num_classes=cfg.num_classes,
                                align_corners=False,
                                layer_idx=-1) if cfg.num_classes > 0 else nn.Identity()
        
        self.loss_func = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index) if cfg.loss_func is None else cfg.loss_func
        
    def forward(self, images, labels=None):
        output = self.segformer(images)
        logits = self.head(output)
        if labels is not None and self.training:
            x = F.interpolate(logits, labels.shape[1:], mode='bilinear')
            losses = self.loss_func(x, labels)
            return {"losses": losses}
        else:
            return {"prediction_scores": logits}
        
    @staticmethod
    def set_pipeline_stage_id(model):
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

if __name__ == '__main__':
    from projects.SegFormer.configs.models.mit_b0 import cfg
    cfg.pretrained_model_path = '/home/zhangguangjun/libai/projects/SegFormer/pretrained'
    print(cfg)
    
    model = SegformerSegmentationLoadImageNetPretrain(cfg)
    
    import numpy as np
    input = np.random.rand(1, 3, 224, 224)
    input = flow.tensor(input, dtype=flow.float32, sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]), placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", [0]),)
    output = model(input)
    print(output['prediction_scores'].shape)