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

from dataclasses import dataclass, field
from typing import List
import oneflow as flow
from libai.utils import distributed as dist


@dataclass
class Metadata:
    tensor: flow.Tensor
    sbp: list = field(default_factory=lambda: dist.get_nd_sbp(
        [flow.sbp.split(0), flow.sbp.broadcast])) 
    placement: flow._oneflow_internal.placement = dist.get_layer_placement(0)
    
    @staticmethod
    def stack(metadata_lists):
        assert len(metadata_lists) > 0
        for data in metadata_lists:
            pass
            

class Instance:
    """ 
    This class represents a instance with metadata as attributes.
    It stores the attributes of an instance (e.g., sbp, placement) as "fields".

    all other (non-filed) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.
    
    Some basic usage:
    
    1. Set/get/check a field:
    
        .. code-block:: python

            instance.sbp = flow.sbp.split(0)
            instance.placement = flow.env.all_device_placement("cpu")
    """
    
    def __init__(self, **kwargs):

        self._fields = {}
        for k, v in kwargs.items():
            self.set(k, v)
    
    def __setattr__(self, name, val):
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val) 
    
    def __getattr__(self, name):
        if name == "_fields" or name not in self._fields:
            raise AttributeError(f"Cannot find field '{name}' in the given Instance!")
        return self._fields[name] 
            
    def set(self, name, value):
        """ 
        Set the field named `name` to `value`.
        """
        self._fields[name] = value
    
    def has(self, name):
        return name in self._fields

    def remove(self, name):
        del self._fields[name]
    
    def get(self, name):
        return self._fields[name]

    def get_fields(self):
        return self._fields
    
    @staticmethod
    def stack(instance_lists: List["Instance"]) -> "Instance":
        assert all(isinstance(i, Instance) for i in instance_lists)
        assert len(instance_lists) > 0

        ret = Instance()
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists] 
            v0 = values[0]
            if not isinstance(v0, flow.Tensor):
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
