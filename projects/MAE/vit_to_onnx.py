from libai.utils.checkpoint import Checkpointer
import oneflow as flow
from oneflow import nn
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check
from projects.MAE.modeling.vit import VisionTransformer

model = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    drop_path_rate=0.1,
    global_pool=True,
)


class VitGraph(nn.Graph):
    def __init__(self, eager_model):
        super().__init__()
        self.model = eager_model

    def build(self, x):
        return self.model(x)


model = Checkpointer(model).load('/model/vit_base')
model.eval()
vit_graph = VitGraph(model)
vit_graph._compile(flow.ones(
    1, 3, 224, 224,
    sbp=flow.sbp.broadcast,
    placement=flow.placement("cuda", ranks=[0]),
))

# 导出为 ONNX 模型并进行检查
convert_to_onnx_and_check(vit_graph,
                          onnx_model_path="configs/",
                          print_outlier=True,
                          dynamic_batch_size=True)
