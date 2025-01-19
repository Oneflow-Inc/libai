import numpy as np
import oneflow as flow
import torch
from segf_loader import SegFLoaderHuggerFace
from transformers import SegformerForSemanticSegmentation as HugSegformerForSemanticSegmentation

from libai.utils import distributed as dist
from projects.SegFormer.configs.models.mit_b0 import cfg as libai_cfg
from projects.SegFormer.modeling.segformer_model import SegformerForSemanticSegmentation

np.random.seed(2022)
input = np.random.rand(1, 3, 512, 512)

input_flow = flow.tensor(input, dtype=flow.float32, sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]), placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", [0]),)
input_torch = torch.tensor(input, dtype=torch.float32)

load_func = SegFLoaderHuggerFace(
    model=SegformerForSemanticSegmentation,
    libai_cfg=libai_cfg,
    pretrained_model_path='/home/zhangguangjun/huggingface_models/segformer'
)


model_libai = load_func.load()
model_libai.eval()

output_libai = model_libai(input_flow)['prediction_scores'].sum()
print(output_libai)

model_torch = HugSegformerForSemanticSegmentation.from_pretrained('/home/zhangguangjun/huggingface_models/segformer')
model_torch.eval()
output_torch = model_torch(input_torch).logits.sum()
print(output_torch)
