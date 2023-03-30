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

# -----------mock torch, put it in the first line-----------
import oneflow as flow

flow.mock_torch.enable(lazy=True)

from oneflow import Tensor, nn  # noqa
from transformers import modeling_utils  # noqa
from transformers.modeling_utils import _load_state_dict_into_model  # noqa


# ---------------- mock _load_state_dict_into_model ------------------
def new_load(model_to_load, state_dict, start_prefix):
    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # TODO: add start_prefix judgement
    for k, v in model_to_load.state_dict().items():
        if k in state_dict and v.is_global:
            state_dict[k] = state_dict[k].to_global(
                sbp=flow.sbp.broadcast, placement=flow.env.all_device_placement("cpu")
            )
            state_dict[k] = state_dict[k].to_global(
                sbp=v.sbp,
                placement=flow.placement("cpu", ranks=list(v.placement.ranks)),
            )

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix.
        # We can exit early if there are none in this state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier.
    # Note that `state_dict` is a copy of the argument, so it's safe to delete it.
    del state_dict
    return error_msgs


modeling_utils._load_state_dict_into_model = new_load


# -----------------mock tensor.new_ones() -------------
def flow_ones(self, *args, **kwargs):
    return flow.ones(*args, **kwargs, device=self.device, dtype=self.dtype)


Tensor.new_ones = flow_ones


# -----------------mock tensor.new() ------------------
def flow_zeros(self, *args, **kwargs):
    return flow.zeros(*args, **kwargs, device=self.device, dtype=self.dtype)


Tensor.new = flow_zeros

# ------------------mock nn.functional.softmax---------
temp_func = nn.functional.softmax


def flow_softmax(*args, **kwargs):
    if "dtype" in kwargs:
        _tensor = args[0].to(dtype=kwargs.pop("dtype"))
        return temp_func(_tensor, *args[1:], **kwargs)
    else:
        return temp_func(*args, **kwargs)


nn.functional.softmax = flow_softmax
