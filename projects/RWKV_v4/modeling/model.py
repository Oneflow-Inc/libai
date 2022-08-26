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
import math
import os

import oneflow as flow
import oneflow.nn.functional as F
from oneflow import nn

from libai.layers import LayerNorm, Linear, VocabEmbedding

logger = logging.getLogger(__name__)

RWKV_HEAD_QK_DIM = 256


class RWKV_TimeMix(nn.Module):
    def __init__(self, vocab_size, ctx_len, model_type, n_layer, n_embd, layer_id):

        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = ctx_len
        self.n_embd = n_embd

        attn_sz = n_embd

        with flow.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0

            # fancy time_decay
            decay_speed = flow.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            # fancy time_first
            zigzag = flow.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5
            self.time_first = nn.Parameter(flow.ones(attn_sz) * math.log(0.3) + zigzag)

            # fancy time_mix
            x = flow.ones(1, 1, n_embd)
            for i in range(n_embd):
                x[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(flow.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(flow.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(flow.pow(x, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = Linear(n_embd, attn_sz, bias=False, parallel="col")
        self.value = Linear(n_embd, attn_sz, bias=False, parallel="col")
        self.receptance = Linear(n_embd, attn_sz, bias=False, parallel="col")

        self.output = Linear(attn_sz, n_embd, bias=False, parallel="row")

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()  # x = (Batch,Time,Channel)

        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x)  # self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        rwkv = flow.sigmoid(r) * flow._C.wkv(B, T, C, self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)

        return rwkv


class RWKV_ChannelMix(nn.Module):
    def __init__(self, vocab_size, ctx_len, model_type, n_layer, n_embd, layer_id):

        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with flow.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0

            x = flow.ones(1, 1, n_embd)
            for i in range(n_embd):
                x[0, 0, i] = i / n_embd

            self.time_mix_k = nn.Parameter(flow.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(flow.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * n_embd
        self.key = Linear(n_embd, hidden_sz, bias=False, parallel="col")
        self.receptance = Linear(n_embd, n_embd, bias=False, parallel="col")
        self.value = Linear(hidden_sz, n_embd, bias=False, parallel="row")

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = flow.square(flow.relu(k))
        kv = self.value(k)

        rkv = flow.sigmoid(self.receptance(xr)) * kv
        return rkv


##########################################################################
# The GPT Model with our blocks
##########################################################################


class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k, v in kwargs.items():
            setattr(self, k, v)


class Block(nn.Module):
    def __init__(self, vocab_size, ctx_len, model_type, n_layer, n_embd, layer_id):
        super().__init__()

        self.layer_id = layer_id

        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = LayerNorm(n_embd)

        # if self.layer_id == 0 and self.model_type == 'RWKV-ffnPre':
        #     self.ffnPre = RWKV_ChannelMix(
        #                         vocab_size,
        #                         ctx_len,model_type,
        #                         n_layer,n_embd,
        #                         layer_id+1000
        #                     )
        # else:
        self.att = RWKV_TimeMix(vocab_size, ctx_len, model_type, n_layer, n_embd, layer_id)

        self.ffn = RWKV_ChannelMix(vocab_size, ctx_len, model_type, n_layer, n_embd, layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, ctx_len, model_type, n_layer, n_embd):

        super().__init__()
        self.step = 0

        self.emb = VocabEmbedding(vocab_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(vocab_size, ctx_len, model_type, n_layer, n_embd, i) for i in range(n_layer)]
        )

        self.ln_out = LayerNorm(n_embd)
        self.head = Linear(n_embd, vocab_size, bias=False, parallel="row")

        if RWKV_HEAD_QK_DIM > 0:
            self.head_q = Linear(n_embd, RWKV_HEAD_QK_DIM, bias=False, parallel="col")
            self.head_q.scale_init = 0
            self.head_k = Linear(n_embd, RWKV_HEAD_QK_DIM, bias=False, parallel="col")
            self.head_k.scale_init = 0.1
            self.register_buffer("copy_mask", flow.tril(flow.ones(ctx_len, ctx_len)))

        self.ctx_len = ctx_len

        try:
            if os.environ["RWKV_LOAD_MODEL"] == str(False):
                RWKV_Init(self, vocab_size, ctx_len, model_type, n_layer, n_embd)  # noqa
        except:  # noqa
            pass

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_ctx_len(self):
        return self.ctx_len

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, idx, targets=None):

        idx = idx.to_global(placement=self.emb.weight.placement)

        self.step += 1
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)

        if RWKV_HEAD_QK_DIM > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
            c = c.float()
            c = c @ F.one_hot(idx, num_classes=6064).float()
            # https://github.com/Chenqll/libai/pull/1#issuecomment-1193328369
            x = self.head(x) + c
        else:
            x = self.head(x)

        if self.training and targets is not None:
            loss = F.cross_entropy(
                x.view(-1, x.size(-1)), targets.to_global(placement=x.placement).view(-1)
            )
            return {"loss": loss}
        else:
            return {"x": x}

    def set_activation_checkpoint(self):
        for module_block in self.modules():
            if isinstance(module_block.origin, Block):
                module_block.config.activation_checkpointing = True
