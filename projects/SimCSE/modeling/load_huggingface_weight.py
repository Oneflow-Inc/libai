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

from collections import OrderedDict

import oneflow as flow
import torch

import libai.utils.distributed as dist


def convert_tensor(tensor):
    tensor = tensor.float()
    return flow.Tensor(tensor.cpu().numpy())


def conver_state(state, layers, hidden_size, num_heads, head_size):
    save = OrderedDict()
    not_saved = []
    Layers = layers
    for name, tensor in state.items():
        if "embeddings" in name:
            if "word_embeddings" in name:
                save["embeddings.vocab_embeddings.weight"] = convert_tensor(tensor)
            elif "position_embeddings" in name:
                save["embeddings.position_embeddings.weight"] = convert_tensor(tensor)
            elif "token_type_embeddings" in name:
                save["embeddings.tokentype_embeddings.weight"] = convert_tensor(tensor)
            elif "LayerNorm.gamma" in name:
                save["encoders.0.input_layernorm.weight"] = convert_tensor(tensor)
            elif "LayerNorm.beta" in name:
                save["encoders.0.input_layernorm.bias"] = convert_tensor(tensor)

        elif "attention" in name:
            if "self" in name:
                index = name.split(".")[3]
                if "encoders." + index + ".self_attention.query_key_value.weight" in save.keys():
                    continue
                q_w = name.replace(name.split(".")[6], "query").replace(
                    name.split(".")[7], "weight"
                )
                k_w = name.replace(name.split(".")[6], "key").replace(name.split(".")[7], "weight")
                v_w = name.replace(name.split(".")[6], "value").replace(
                    name.split(".")[7], "weight"
                )
                q_b = name.replace(name.split(".")[6], "query").replace(name.split(".")[7], "bias")
                k_b = name.replace(name.split(".")[6], "key").replace(name.split(".")[7], "bias")
                v_b = name.replace(name.split(".")[6], "value").replace(name.split(".")[7], "bias")

                qkv_w = torch.cat((state[q_w], state[k_w], state[v_w]), dim=0)  # 【768*3， 768】
                # function for weight-----------------------------------
                qkv_w = qkv_w.view([3, num_heads, head_size, hidden_size])
                qkv_w = qkv_w.permute(1, 0, 2, 3).contiguous().view(3 * hidden_size, hidden_size)
                # ---------------------------------------------------------

                qkv_b = torch.cat((state[q_b], state[k_b], state[v_b]), dim=-1)
                # function for bias--------------------------------------
                qkv_b = qkv_b.view(3, num_heads, head_size)
                qkv_b = qkv_b.permute(1, 0, 2).contiguous().view(-1)
                # ---------------------------------------------------------

                target_w = "encoders." + index + ".self_attention.query_key_value.weight"
                save[target_w] = convert_tensor(qkv_w)
                target_b = "encoders." + index + ".self_attention.query_key_value.bias"
                save[target_b] = convert_tensor(qkv_b)
            elif "output" in name:
                index = name.split(".")[3]
                if "dense" in name:
                    if "weight" in name:
                        target = "encoders." + index + ".self_attention.dense.weight"
                        save[target] = convert_tensor(tensor)
                    elif "bias" in name:
                        target = "encoders." + index + ".self_attention.dense.bias"
                        save[target] = convert_tensor(tensor)
                elif "LayerNorm" in name:
                    if "gamma" in name:
                        target = "encoders." + index + ".post_attention_layernorm.weight"
                        save[target] = convert_tensor(tensor)
                    elif "beta" in name:
                        target = "encoders." + index + ".post_attention_layernorm.bias"
                        save[target] = convert_tensor(tensor)

        elif "intermediate" in name:
            index = name.split(".")[3]
            if "encoders." + index + ".mlp.dense_h_to_4h.weight" in save.keys():
                continue
            w = "bert.encoder.layer." + index + ".intermediate.dense.weight"
            b = "bert.encoder.layer." + index + ".intermediate.dense.bias"
            t_w = "encoders." + index + ".mlp.dense_h_to_4h.weight"
            t_b = "encoders." + index + ".mlp.dense_h_to_4h.bias"
            save[t_w] = convert_tensor(state[w])
            save[t_b] = convert_tensor(state[b])

        elif "output" in name:
            index = name.split(".")[3]
            if "dense.weight" in name:
                target = "encoders." + index + ".mlp.dense_4h_to_h.weight"
                save[target] = convert_tensor(tensor)
            elif "dense.bias" in name:
                target = "encoders." + index + ".mlp.dense_4h_to_h.bias"
                save[target] = convert_tensor(tensor)
            elif "LayerNorm.gamma" in name:
                if index == str(Layers - 1):
                    target = "final_layernorm.weight"
                    save[target] = convert_tensor(tensor)
                    continue
                target = "encoders." + str(int(index) + 1) + ".input_layernorm.weight"
                save[target] = convert_tensor(tensor)
            elif "LayerNorm.beta" in name:
                if index == str(Layers - 1):
                    target = "final_layernorm.bias"
                    save[target] = convert_tensor(tensor)
                    continue
                target = "encoders." + str(int(index) + 1) + ".input_layernorm.bias"
                save[target] = convert_tensor(tensor)

        elif "pooler" in name:
            if "weight" in name:
                save["pooler.dense.weight"] = convert_tensor(tensor)
            elif "bias" in name:
                save["pooler.dense.bias"] = convert_tensor(tensor)
        else:
            not_saved.append(name)
    return save, not_saved


def load_tensor(tensor_lhs, tensor_rhs):
    tensor_rhs = flow.to_global(
        tensor_rhs,
        placement=tensor_lhs.placement,
        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
    )
    tensor_rhs = tensor_rhs.to_global(sbp=tensor_lhs.sbp)
    tensor_lhs.copy_(tensor_rhs)


def load_huggingface_bert(model, path, hidden_size, num_heads, layers=12):
    head_size = hidden_size // num_heads
    huggingface_state_dict = torch.load(path)
    of_state_dict, _ = conver_state(
        huggingface_state_dict,
        layers=layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_size=head_size,
    )
    for key, value in of_state_dict.items():
        load_tensor(model.state_dict()[key], value)
