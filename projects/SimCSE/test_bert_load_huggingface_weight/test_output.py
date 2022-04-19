import numpy as np
import oneflow as flow
import torch
import transformers
from load_huggingface_weight import load_huggingface_bert

import libai
from libai.config import LazyCall
from libai.models import build_model
from libai.utils import distributed as dist

input_ids = [[101, 1962, 2110, 739, 999, 1, 2, 3, 102]]
mask = [[1] * len(input_ids)]

# libai's Bert
cfg = dict(
    vocab_size=21128,
    hidden_size=768,
    hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    num_tokentypes=2,
    add_pooling_layer=True,
    initializer_range=0.02,
    layernorm_eps=1e-12,
    bias_gelu_fusion=False,
    bias_dropout_fusion=False,
    scale_mask_softmax_fusion=False,
    apply_query_key_layer_scaling=False,
    add_binary_head=True,
    amp_enabled=False,
    apply_residual_post_layernorm=True,
)
bert_lib = build_model(LazyCall(libai.models.BertModel)(cfg=cfg))
load_huggingface_bert(
    bert_lib,
    "./bert-base-chinese/pytorch_model.bin",
    cfg["hidden_size"],
    cfg["num_attention_heads"],
)
input_of = flow.tensor(
    input_ids,
    dtype=flow.long,
    sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
    placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", [0]),
)
mask_of = flow.tensor(
    mask,
    dtype=flow.long,
    sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
    placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", [0]),
)
bert_lib.eval()
last_hidden_state_of, pooler_output_of = bert_lib(input_of, mask_of)


# huggingface's Bert
bert_hug = transformers.BertModel.from_pretrained("./bert-base-chinese")
bert_hug.eval()
input_pt = torch.tensor(input_ids)
mask_pt = torch.tensor(mask)
last_hidden_state_pt = bert_hug(input_pt, mask_pt).last_hidden_state

res1 = last_hidden_state_of.detach().numpy()
res2 = last_hidden_state_pt.detach().numpy()

print(res1.sum())
print(res2.sum())