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

from libai.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

logger = logging.getLogger(__name__)


def convert_and_copy_tensor(tensor_lhs: flow.Tensor, tensor_rhs: torch.Tensor):
    """ copy torch tensor weights to flow tensor weights

    Args:
        tensor_lhs (flow.Tensor)
        tensor_rhs (torch.Tensor)
    """
    tensor_rhs = flow.Tensor(tensor_rhs.cpu().float().numpy())
    tensor_rhs = flow.to_global(tensor_rhs, placement=tensor_lhs.placement, sbp=tensor_lhs.sbp)
    tensor_lhs.copy_(tensor_rhs)


def load_megatron_gpt(model: flow.nn.Module, model_weight_path: torch.nn.Module):
    import torch

    logger.info("Loading megatron gpt weight")
    model_weight_state_dict = torch.load(model_weight_path, map_location="cpu")
    flow_state_dict = model.state_dict()


    print('flow_weight nums: ', len(flow_state_dict))
    print('torch_weight nums: ', len(model_weight_state_dict))

    used_flow_keys = set()
    used_torch_keys = set()

    for torch_k, v in model_weight_state_dict.items():
        k = torch_k
        if 'embedding' not in k and '.weight'in k and len(v.shape) == 2 and 'embedding.' not in k:
            v = v.transpose(0, 1)
        if k == 'language_model.embedding.word_embeddings.weight':
            k = 'embeddings.token_embeddings.weight'
        elif k == 'language_model.embedding.position_embeddings.weight':
            k = 'embeddings.position_embeddings.weight'
        elif k.startswith('language_model.encoder.final_layernorm.'):
            k = k.replace('language_model.encoder.final_layernorm', 'transformer.layernorm_f')
        elif k.startswith('language_model.encoder'):
            k = k.replace('language_model.encoder', 'transformer')

        convert_and_copy_tensor(flow_state_dict[k], v)
        used_flow_keys.add(k)
        used_torch_keys.add(torch_k)

    assert len(set(flow_state_dict.keys()) - used_flow_keys) == 0
    assert len(set(model_weight_state_dict.keys()) - used_torch_keys) == 0
        