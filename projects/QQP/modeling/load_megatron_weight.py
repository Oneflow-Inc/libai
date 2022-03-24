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

import oneflow as flow
import torch

import libai.utils.distributed as dist
from libai.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

logger = logging.getLogger("libai." + __name__)


def convert_tensor(tensor: torch.Tensor):
    tensor = tensor.float()
    return flow.Tensor(tensor.cpu().numpy())


def change_megatron_key(state_dict):
    of_state_dict = {}

    # Language model.
    language_model = state_dict["language_model"]

    # Embedding.
    embedding = language_model["embedding"]
    of_state_dict["embeddings.vocab_embeddings.weight"] = convert_tensor(
        embedding["word_embeddings"]["weight"]
    )
    of_state_dict["embeddings.position_embeddings.weight"] = convert_tensor(
        embedding["position_embeddings"]["weight"]
    )
    of_state_dict["embeddings.tokentype_embeddings.weight"] = convert_tensor(
        embedding["tokentype_embeddings"]["weight"]
    )

    # Encoder.
    encoder = language_model["encoder"]
    for key, value in encoder.items():
        # Change layers.0.input_layernorm.weight -> encoder.layers_0.input_layernorm.weight
        key = "encoders." + key.replace("layers.", "")
        if key.startswith("encoders.final_layernorm"):
            key = key.replace("encoders.", "")
        of_state_dict[key] = convert_tensor(value)

    # Pooler.
    pooler = language_model["pooler"]
    of_state_dict["pooler.dense.weight"] = convert_tensor(pooler["dense.weight"])
    of_state_dict["pooler.dense.bias"] = convert_tensor(pooler["dense.bias"])

    # LM head.
    lm_head = state_dict["lm_head"]
    of_state_dict["cls.predictions.dense.weight"] = convert_tensor(lm_head["dense.weight"])
    of_state_dict["cls.predictions.dense.bias"] = convert_tensor(lm_head["dense.bias"])

    of_state_dict["cls.predictions.layernorm.weight"] = convert_tensor(lm_head["layernorm.weight"])
    of_state_dict["cls.predictions.layernorm.bias"] = convert_tensor(lm_head["layernorm.bias"])

    of_state_dict["lm_logits.bias"] = convert_tensor(lm_head["bias"])

    # Binary head.
    binary_head = state_dict["binary_head"]
    of_state_dict["cls.seq_relationship.weight"] = convert_tensor(binary_head["weight"])
    of_state_dict["cls.seq_relationship.bias"] = convert_tensor((binary_head["bias"]))

    return of_state_dict


def load_tensor(tensor_lhs, tensor_rhs):
    tensor_rhs = flow.to_global(
        tensor_rhs,
        placement=tensor_lhs.placement,
        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
    )
    tensor_rhs = tensor_rhs.to_global(sbp=tensor_lhs.sbp)
    tensor_lhs.copy_(tensor_rhs)


def load_model(model: flow.nn.Module, state_dict):
    model_state_dict = model.state_dict()

    # Decide shape
    incorrect_shapes = []
    for k in list(state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_ckpt = tuple(state_dict[k].shape)
            if shape_model != shape_ckpt:
                incorrect_shapes.append((k, shape_ckpt, shape_model))
                state_dict.pop(k)

    unexpected_keys = []
    for key, value in state_dict.items():
        if key not in model_state_dict:
            unexpected_keys.append(key)
            continue
        model_state_dict.pop(key)
        load_tensor(model.state_dict()[key], value)

    missing_keys = list(model_state_dict.keys())

    for k, shape_checkpoint, shape_model in incorrect_shapes:
        logger.warning(
            "Skip loading parameter '{}' to the model due to incompatible "
            "shapes: {} in the checkpoint but {} in the "
            "model! You might want to double check if this is expected.".format(
                k, shape_checkpoint, shape_model
            )
        )
    if missing_keys:
        logger.info(get_missing_parameters_message(missing_keys))
    if unexpected_keys:
        logger.info(get_unexpected_parameters_message(unexpected_keys))


def load_megatron_bert(model: flow.nn.Module, model_weight_path: str):
    import torch

    megatron_state_dict = torch.load(model_weight_path, map_location="cpu")["model"]
    of_state_dict = change_megatron_key(megatron_state_dict)
    load_model(model, of_state_dict)
