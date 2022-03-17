from libai.config import LazyCall
from projects.MOCOV3.modeling.vit_moco import VisionTransformerMoCo
from utils.weight_convert_tools import load_torch_checkpoint
import oneflow as flow

# freeze all layers but the last head
# print("freeze all layers but the last head")
# linear_keyword = "head"
# for name, param in VisionTransformerMoCo().named_parameters():
#     if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
#         param.requires_grad = False

# # init the head layer
# getattr(VisionTransformerMoCo(), linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
# getattr(VisionTransformerMoCo(), linear_keyword).bias.data.zeros_()

# VisionTransformerMoCo = load_torch_checkpoint(VisionTransformerMoCo(),"projects/MOCOV3/output/vit-s-300ep.pth.tar")

model = LazyCall(VisionTransformerMoCo)(
    
)