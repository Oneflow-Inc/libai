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

from utils.weight_convert import load_torch_checkpoint_linear_prob

from libai.utils.checkpoint import (
    Checkpointer,
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)

logger = logging.getLogger("libai." + __name__)


def load_checkpoint(model, path, weight_style, num_heads, embed_dim):
    linear_keyword = "head"
    for name, param in model.named_parameters():
        if name not in ["%s.weight" % linear_keyword, "%s.bias" % linear_keyword]:
            param.requires_grad = False
    assert weight_style in ["pytorch", "oneflow"]
    if weight_style == "pytorch":
        params = load_torch_checkpoint_linear_prob(num_heads, embed_dim, path=path)
    else:
        params = Checkpointer(model).load(path)

    model_state_dict = model.state_dict()

    # check the incorrect shape and unexpected keys
    incorrect_shapes = []
    unexpected_keys = []
    for k in list(params.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_ckp = tuple(params[k].shape)
            if shape_model != shape_ckp:
                incorrect_shapes.append((k, shape_ckp, shape_model))
                params.pop(k)
            model_state_dict.pop(k)
        else:
            unexpected_keys.append(k)

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

    model.load_state_dict(params, strict=False)
