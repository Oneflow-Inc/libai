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
from oneflow import nn

from libai.utils import distributed as dist


class Conv1D(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        parallel="data",
        init_method=nn.init.xavier_normal_,
        skip_bias_add=False,
        dtype=flow.float32,
        *,
        layer_idx=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.parallel = parallel
        self.skip_bias_add = skip_bias_add

        if parallel == "col":
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)])
            bias_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])

        elif parallel == "row":
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])
            bias_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])

        elif parallel == "data":
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
            bias_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])

        else:
            raise KeyError(f"{parallel} is not supported! Only support ('data', 'row' and 'col')")

        self.weight = flow.nn.Parameter(
            flow.empty(
                (in_features, out_features),
                dtype=dtype,
                placement=dist.get_layer_placement(layer_idx),  # for pipeline parallelism placement
                sbp=weight_sbp,
            )
        )
        if os.getenv("ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT", "0") != "1":
            init_method(self.weight)

        self.bias = (
            flow.nn.Parameter(
                flow.zeros(
                    (out_features,),
                    dtype=dtype,
                    placement=dist.get_layer_placement(layer_idx),
                    sbp=bias_sbp,
                )
            )
            if bias
            else None
        )

    def forward(self, x):
        if dist.same_sbp(self.weight.sbp, dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)])):
            if self.weight.sbp[-1] == flow.sbp.split(1):
                x_sbp = x.sbp[:-1] + (flow.sbp.broadcast,)
                x = x.to_global(sbp=x_sbp)

            x = x.to_global(grad_sbp=x.sbp)
            x = flow.matmul(x, self.weight)

        elif dist.same_sbp(
            self.weight.sbp, dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])
        ):
            if self.weight.sbp[-1] == flow.sbp.split(0):
                x_sbp = x.sbp[:-1] + (flow.sbp.split(x.ndim - 1),)
                x = x.to_global(sbp=x_sbp)
                out_sbp = x.sbp[:-1] + (flow.sbp.broadcast,)
            else:
                out_sbp = x.sbp

            x = flow.matmul(x, self.weight)
            x = x.to_global(sbp=out_sbp)

        elif dist.same_sbp(
            self.weight.sbp, dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        ):
            x = x.to_global(grad_sbp=x.sbp)
            x = flow.matmul(x, self.weight)
        else:
            x = flow.matmul(x, self.weight)

        if self.bias is not None:
            if self.skip_bias_add:
                return x, self.bias
            else:
                return x + self.bias
        else:
            return x

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, parallel={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.parallel,
        )
