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

import oneflow as flow
import sys

sys.path.append(".")

from libai.data import Metadata, Instance

data1 = Metadata(flow.Tensor(3, 3))
data2 = Metadata(flow.Tensor(4, 4))

item1 = Instance(tokens=data1, mask=data2)
item2 = Instance(tokens=data1, mask=data2)

batch_item = Instance.cat(item1, item2)
