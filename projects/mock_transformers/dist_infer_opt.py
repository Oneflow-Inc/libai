import os

os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"
os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"
import init_env  # noqa
from typing import List, Optional, Tuple, Union
import oneflow as flow
import oneflow as torch
import oneflow.nn as nn
from omegaconf import DictConfig
from oneflow.utils.global_view import global_mode
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.opt import modeling_opt

from libai.layers import Linear
from libai.utils import distributed as dist
import numpy as np
import time

# ------replace attention to libai------
temp_class = modeling_opt.OPTAttention


class LiBaiOPTAttention(temp_class):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim = kwargs["embed_dim"]
        bias = kwargs["bias"]
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias, parallel="col", dtype=flow.float16)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias, parallel="col", dtype=flow.float16)
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, parallel="col", dtype=flow.float16)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, parallel="row", dtype=flow.float16)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        fallback = key_value_states is not None or output_attentions or not self.is_decoder
        if fallback:
            return super().forward(
                hidden_states,
                key_value_states,
                past_key_value,
                attention_mask,
                layer_head_mask,
                output_attentions,
            )
        bsz, tgt_len, _ = hidden_states.size()

        query_states, key_states, value_states = flow._C.grouped_matmul_bias(
            [hidden_states, hidden_states, hidden_states],
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias],
        )
        if past_key_value is not None:
            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(value_states, -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(value_states, -1, bsz)

        past_key_value = (key_states, value_states)

        attn_output = flow._C.fused_multi_head_attention_inference_v2(
            query=query_states,
            query_layout="BM(HK)",
            query_head_size=self.head_dim,
            key=key_states,
            key_layout="BHMK",
            value=value_states,
            value_layout="BHMK",
            causal=True,
            causal_diagonal_offset=key_states.shape[2] - query_states.shape[1],
        )
        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value


modeling_opt.OPTAttention = LiBaiOPTAttention

# ----------replace Decoder to libai -----
temp_class = modeling_opt.OPTDecoderLayer


class LiBaiOPTDecoderLayer(temp_class):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = args[0]
        self.fc1 = Linear(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            parallel="col",
            dtype=flow.float16,
        )
        self.fc2 = Linear(
            config.ffn_dim,
            self.embed_dim,
            bias=config.enable_bias,
            parallel="row",
            dtype=flow.float16,
        )


modeling_opt.OPTDecoderLayer = LiBaiOPTDecoderLayer

if __name__ == "__main__":
    # set dist config
    parallel_config = DictConfig(
        dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,  # set to 1, unsupport pipeline parallel now
            pipeline_num_layers=None,
            device_type="cpu",
        )
    )
    dist.setup_dist_util(parallel_config)

    # initial and load model
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b", torch_dtype=flow.float16)
    # set model to cuda
    dist.set_device_type("cuda")
    model._apply(dist.convert_to_distributed_default_setting)
    # initial tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b", use_fast=False)

    # get input_ids
    prompt = "Hello, I'm am conscious and"
    input_ids = tokenizer(prompt, return_tensors="np").input_ids
    input_ids = flow.from_numpy(input_ids)
    input_ids = input_ids.to_global(
        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        placement=dist.get_layer_placement(0),
    )

    # generate id
    placement_sbp_dict = dict(
        placement=flow.env.all_device_placement("cuda"), sbp=flow.sbp.broadcast,
    )
    p = time.time()
    while True:
        with global_mode(True, **placement_sbp_dict):
            generated_ids = model.generate(input_ids, max_length=128, min_length=128)
        out_put_ids = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        t = time.time()
        print(t - p, (t - p) / generated_ids.shape[1], out_put_ids)
        p = t
