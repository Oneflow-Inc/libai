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

import logging
import sys
import numpy as np
import pdb

# 
import oneflow as flow
import torch

#
from libai.engine import DefaultTrainer
from libai.config import LazyConfig
logger = logging.getLogger(__name__)

# 
sys.path.append('/home/chenqiaoling/RWKV-LM/RWKV-v4')
from src.model import GPT, GPTConfig
from modeling.model import GPT as model

def convert_qkv_weight( value):
    """
    Convert qkv.weight to be compatible with LiBai transformer layer

    Args:
        cfg: config file
        value: qkv.weight in the loaded checkpoint
    """
    num_heads = model.num_heads
    hidden_size = model.embed_dim
    head_size = int(hidden_size / num_heads)
    qkv_weight = (
        value.view([3, num_heads, head_size, hidden_size])
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(hidden_size * 3, hidden_size)
    )
    return qkv_weight


def convert_qkv_bias( value):
    """
    Convert qkv.bias to be compatible with LiBai transformer layer

    Args:
        cfg: config file
        value: qkv.bias in the loaded checkpoint
    """
    num_heads = model.num_heads
    hidden_size = model.embed_dim
    head_size = int(hidden_size / num_heads)
    qkv_bias = (
        value.view(3, num_heads, head_size).permute(1, 0, 2).contiguous().view(hidden_size * 3)
    )
    return qkv_bias


def filter_keys(key, value):
    """
    Filtering the state_dict keys and values to match LiBai's MAE model
    """
    if "norm1" in key:
        key = key.replace("norm1", "input_layernorm")
    elif "attn.qkv" in key:
        key = key.replace("attn.qkv", "self_attention.query_key_value")
        if "weight" in key:
            value = convert_qkv_weight( value)
        if "bias" in key:
            value = convert_qkv_bias( value)
    elif "attn.proj" in key:
        key = key.replace("attn.proj", "self_attention.dense")
    elif "norm2" in key:
        key = key.replace("norm2", "post_attention_layernorm")
    elif "mlp.fc1" in key:
        key = key.replace("mlp.fc1", "mlp.dense_h_to_4h")
    elif "mlp.fc2" in key:
        key = key.replace("mlp.fc2", "mlp.dense_4h_to_h")
    elif "fc_norm" in key:
        key = key.replace("fc_norm", "norm")

    return key, value


def load_torch_checkpoint(model,  path="/home/chenqiaoling/RWKV-LM/RWKV-v4/trained-1.pth", strict=False):
    """
    Load checkpoint from the given torch weights.
    Torch weight can be downloaded from the original repo:
        https://github.com/facebookresearch/mae
    """
    torch_dict = torch.load(path, map_location="cpu")
    parameters = torch_dict
    new_parameters = dict()
    for key, value in parameters.items():
        if "num_batches_tracked" not in key:
            # to global tensor
            key, val = filter_keys(key, value)
            val = val.detach().cpu().numpy().astype(np.float32)
            val = flow.tensor(val).to_global(
                sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0])
            )
            new_parameters[key] = val
    return new_parameters
    # model.load_state_dict(new_parameters, strict=strict)
    # print("Successfully load torch mae checkpoint.")
    # return model

# 转换权重
para=dict()
para=load_torch_checkpoint(model)

# 转入config
cfg = LazyConfig.load('/home/chenqiaoling/RWKV-LM/libai/projects/RWKV_V4/configs/config.py')
model_oneflow = DefaultTrainer.build_model(cfg)

model_oneflow.load_state_dict(para, strict=False)

# 定义数据，需要对齐 shape 和 dtype
input = torch.rand(12,1024)
input = input.detach().cpu().numpy().astype(np.int64)

# 将数据转换为 oneflow 的 tensor
input_flow = flow.tensor(input).to_global(
    sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0])
    )

# 将数据转换为 torch 的 tensor
input=torch.tensor(input)

# 引入 torch 的 model
model = GPT(GPTConfig(6064, 1024, model_type='RWKV',
                        n_layer=6, n_embd=512))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device) # 移动模型到cuda
input=input.to(device)
m=torch.load('/home/chenqiaoling/RWKV-LM/RWKV-v4/trained-1.pth',map_location=torch.device('cpu'))
model.load_state_dict(m)

# 在 flow.no_grad() 下进行前向，得到输出
with flow.no_grad():
    out_flow=model_oneflow(input_flow)
    out_torch=model(input)
pdb.set_trace()

out1=out_flow['x'].numpy()
out2=out_torch['x'].detach().cpu().numpy()

print(np.allclose(out1, out2, 1e-4, 1e-4))
# 进行模型输出验证
# dict1=set(list(out_flow))
# dict2=set(list(out_torch))
# print(dict1-dict2)
# print(dict2-dict1)
# pdb.set_trace()

# print('*********')
# print(para.keys())
# print('*********')
# print(model_oneflow.state_dict().keys())

# dict1=set(list(para.keys()))
# dict2=set(list(model_oneflow.state_dict().keys()))
# 先初始化数据 -- 确保和 model 的 forward 内要求的数据（如 idx）的 shape 一致
# 分别转为 pytorch 和 oneflow 的 tensor 输入 model 进行前向 可以仿照/home/chenqiaoling/RWKV-LM/RWKV-v4/src/trainer.py 的 L121 进行前向
