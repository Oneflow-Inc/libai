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

import os

import oneflow as flow

from libai.utils import distributed as dist

flow.mock_torch.enable()

from transformers import BertTokenizer, GPT2Tokenizer, MT5Tokenizer, T5Tokenizer  # noqa
from transformers.tokenization_utils_base import *  # noqa
from transformers.utils import generic  # noqa
from transformers.utils.generic import TensorType  # noqa


# ---------------- mock TensorType ------------------
class TensorType(ExplicitEnum):  # noqa
    PYTORCH = "pt"
    TENSORFLOW = "tf"
    ONEFLOW = "of"
    NUMPY = "np"
    JAX = "jax"


generic.TensorType = TensorType


# ---------------- mock convert_to_tensors ------------------
def flow_convert_to_tensors(self, tensor_type=None, prepend_batch_axis=False):
    if tensor_type is None:
        return self

    # Convert to TensorType
    if not isinstance(tensor_type, TensorType):
        tensor_type = TensorType(tensor_type)
    as_tensor = None
    is_tensor = None
    # Get a function reference for the correct framework
    if tensor_type == TensorType.TENSORFLOW:
        if not is_tf_available():  # noqa
            raise ImportError(
                "Unable to convert output to TensorFlow tensors format, TensorFlow is not "
                "installed."
            )
        import tensorflow as tf

        as_tensor = tf.constant
        is_tensor = tf.is_tensor
    elif tensor_type == TensorType.PYTORCH:
        if not is_torch_available():  # noqa
            raise ImportError(
                "Unable to convert output to PyTorch tensors format, PyTorch is not installed."
            )
        import torch

        as_tensor = torch.tensor
        is_tensor = torch.is_tensor
    elif tensor_type == TensorType.ONEFLOW:
        try:
            import oneflow  # noqa
        except ImportError as e:
            msg = "Unable to convert output to OneFlow tensors format, OneFlow is not installed."
            raise ImportError(msg) from e
        as_tensor = flow.tensor
        is_tensor = flow.is_tensor
    elif tensor_type == TensorType.JAX:
        if not is_flax_available():  # noqa
            raise ImportError(
                "Unable to convert output to JAX tensors format, JAX is not installed."
            )
        import jax.numpy as jnp  # noqa: F811

        as_tensor = jnp.array
        is_tensor = is_jax_tensor  # noqa
    else:
        as_tensor = np.asarray  # noqa
        is_tensor = is_numpy_array  # noqa

    # Do the tensor conversion in batch
    for key, value in self.items():
        try:
            if prepend_batch_axis:
                value = [value]

            if not is_tensor(value):
                tensor = as_tensor(value)

                # Removing this for now in favor of controlling the shape with `prepend_batch_axis`
                # # at-least2d
                # if tensor.ndim > 2:
                #     tensor = tensor.squeeze(0)
                # elif tensor.ndim < 2:
                #     tensor = tensor[None, :]

                self[key] = tensor
        except Exception as e:
            if key == "overflowing_tokens":
                raise ValueError(
                    "Unable to create tensor returning overflowing tokens of different lengths. "
                    "Please see if a fast version of this tokenizer is available to have this "
                    "feature available."
                ) from e
            raise ValueError(
                "Unable to create tensor, you should probably activate truncation and/or "
                "padding with 'padding=True' 'truncation=True' to have batched tensors with "
                f"the same length. Perhaps your features (`{key}` in this case) have "
                "excessive nesting (inputs type `list` where type `int` is expected)."
            ) from e
    if os.getenv("IS_GLOBAL", True) is True:
        size = self["input_ids"].size()
        sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])

        for k, v in self.items():
            if is_tensor != flow.is_tensor:
                raise ValueError(
                    "Unable to create tensor, you should probably set `return_tensors='of'` "
                )
            if v.size() != size:
                raise ValueError(
                    "Unable to create tensor, you should probably padding with `padding=True` "
                )
            self[k] = v.to_global(sbp=sbp, placement=dist.get_layer_placement(0))
    return self


BatchEncoding.convert_to_tensors = flow_convert_to_tensors  # noqa
