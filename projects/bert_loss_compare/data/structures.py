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
from dataclasses import dataclass, field
from typing import Any, List

import oneflow as flow

from libai.utils import distributed as dist


@dataclass
class DistTensorData:
    tensor: flow.Tensor
    sbp_list: list = field(default_factory=lambda: ["split_0", "broadcast"])
    placement_idx: int = 0

    # Tensor-like methods
    def to_consistent(self, sbp=None, placement=None):
        if sbp is not None:
            self.sbp = sbp
        else:
            sbp_list = []
            for sbp in self.sbp_list:
                sbp = sbp.split("_")
                if len(sbp) > 1:
                    # split dim
                    assert sbp[0] == "split"
                    split_dim = int(sbp[1])
                    sbp_list.append(flow.sbp.split(split_dim))
                else:
                    sbp_sign = sbp[0]
                    sbp_list.append(getattr(flow.sbp, sbp_sign))
            self.sbp = dist.get_nd_sbp(sbp_list)

        if placement is not None:
            self.placement = placement
        else:
            self.placement = dist.get_layer_placement(self.placement_idx)

        self.tensor = self.tensor.to_consistent(sbp=self.sbp, placement=self.placement)

    @staticmethod
    def stack(distTensor_lists: List["DistTensorData"]) -> "DistTensorData":
        if not isinstance(distTensor_lists[0].tensor, flow.Tensor):
            raise TypeError(
                "DistTensorData.tensor must be a flow.Tensor, but got {}. "
                "Please check the return values of `__getitem__` in dataset.".format(
                    type(distTensor_lists[0].tensor)
                )
            )

        assert len(distTensor_lists) > 0
        if len(distTensor_lists) == 1:
            # TODO(l1aoxingyu): add inplace unsqueeze
            # distTensor_lists[0].tensor.unsqueeze_(0)  # add batch dim
            distTensor_lists[0].tensor = distTensor_lists[0].tensor.unsqueeze(0)  # add batch dim
            return distTensor_lists[0]

        tensor_size = distTensor_lists[0].tensor.size()
        sbp_list = distTensor_lists[0].sbp_list
        placement_idx = distTensor_lists[0].placement_idx
        tensors = []
        for data in distTensor_lists:
            assert (
                data.tensor.size() == tensor_size
            ), f"tensor shape is not equal, {data.tensor.size()} != {tensor_size}"
            assert (
                data.sbp_list == sbp_list
            ), f"sbp_list is not equal, {data.sbp_list} != {sbp_list}!"
            assert (
                data.placement_idx == placement_idx
            ), f"placement_idx is not equal, {data.placement_idx} != {placement_idx}"
            tensors.append(data.tensor)
        tensors = flow.stack(tensors, dim=0)
        ret = DistTensorData(tensors, sbp_list=sbp_list, placement_idx=placement_idx)
        return ret


class Instance:
    """
    This class represents a instance with metadata as attributes.
    It stores the attributes of an instance (e.g., image, tokens) as "fields".

    all other (non-filed) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

        .. code-block:: python

            instance.tokens = Metadata(...)
            instance.mask = Metadata(...)
            print(instance.tokens)
            print(instance.has("mask")) # True

    2. ``len(instance)`` returns the number of instance
    """

    def __init__(self, **kwargs):

        self._fields = OrderedDict()
        for k, v in kwargs.items():
            self.set(k, v)

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str):
        if name == "_fields" or name not in self._fields:
            raise AttributeError(f"Cannot find field '{name}' in the given Instance!")
        return self._fields[name]

    def set(self, name: str, value: Any):
        """
        Set the field named `name` to `value`.
        """
        self._fields[name] = value

    def has(self, name: str):
        return name in self._fields

    def remove(self, name: str):
        del self._fields[name]

    def get(self, name: str):
        return self._fields[name]

    def get_fields(self):
        return self._fields

    def __len__(self):
        return len(self._fields.keys())

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def stack(instance_lists: List["Instance"]) -> "Instance":
        assert all(isinstance(i, Instance) for i in instance_lists)
        assert len(instance_lists) > 0

        ret = Instance()
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, flow.Tensor):
                values = flow.stack(values, dim=0)
            elif isinstance(v0, list):
                pass
            elif hasattr(type(v0), "stack"):
                values = type(v0).stack(values)
            else:
                raise ValueError("Unsupported type {} for stack.".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self):
        s = self.__class__.__name__ + "("
        s += "fields=[{}]".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__
