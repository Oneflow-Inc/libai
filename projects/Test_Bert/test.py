from load_huggingface_weight import load_huggingface_bert

import oneflow as flow
import libai
from libai.models import build_model
from libai.config import LazyCall
from libai.utils import distributed as dist

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
    bias_gelu_fusion=False, #
    bias_dropout_fusion=False,#
    scale_mask_softmax_fusion=False,
    apply_query_key_layer_scaling=False,#
    add_binary_head=True,
    amp_enabled=False,
)

bert = build_model(LazyCall(libai.models.BertModel)(cfg=cfg))
load_huggingface_bert(bert, './pretrain/pytorch_model.bin')

input_ids = [[101, 1962, 2110, 739, 102]]
mask = [[1]*len(input_ids)]

input_of = flow.tensor(input_ids, dtype=flow.long, sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]), placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", [0]),)
mask_of = flow.tensor(mask, dtype=flow.long, sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]), placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", [0]),)

bert.eval()
last_hidden_state_of, pooler_output_of = bert(input_of, mask_of)

print(last_hidden_state_of)
