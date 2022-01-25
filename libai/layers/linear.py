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
from oneflow import nn

from libai.utils import distributed as dist


class Linear1D(nn.Module):
    """Linear layer with 1D parallelism which includes column parallelism and row parallelism.
    The linear layer is defined as :math:`Y = XA + b`.

    In column parallelism, A is parallelized along the second dimension
    as :math:`A = [A_1, ..., A_p]`.

    In row parallelism, A is parallelized along the first dimension and X along its second
    dimension as:
                | A_1 |
                |  .  |
            A = |  .  |         X = [X_1, ..., X_p]
                |  .  |
                | A_p |

    Arguments:
        in_features: size of each input sample.
        out_features: size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias. Defaults to ``True``.
        parallel: . Defaults to "data".
        init_method: method to initialize weight. Defaults to nn.init.xavier_normal_.
        skip_bias_add: skip adding bias but instead return it, so that adding bias can be fused with
        other elementwise operations. Defaults to ``False``.
        layer_idx: A layer_idx sign which determines the placement. It will be used in pipeline
        parallelism. Defaults to 0.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        parallel="data",
        init_method=nn.init.xavier_normal_,
        skip_bias_add=False,
        *,
        layer_idx=0,  # enforce layer_idx passed with keyword
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.parallel = parallel
        self.skip_bias_add = skip_bias_add

        if parallel == "col":
            # Column parallel weight sbp: [B, S(1)] and bias sbp: [B, S(0)].
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)])
            bias_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])
        elif parallel == "row":
            # Row parallel weight sbp: [B, S(0)] and bias sbp: [B, B]
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])
            bias_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        elif parallel == "data":
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
            bias_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        else:
            raise KeyError(f"{parallel} is not supported! Only support ('data', 'row' and 'col')")

        self.weight = flow.nn.Parameter(
            flow.empty(
                (in_features, out_features),
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),  # for pipeline parallelism placement
                sbp=weight_sbp,
            )
        )
        init_method(self.weight)

        self.bias = (
            flow.nn.Parameter(
                flow.zeros(
                    (out_features,),
                    dtype=flow.float32,
                    placement=dist.get_layer_placement(layer_idx),
                    sbp=bias_sbp,
                )
            )
            if bias
            else None
        )

    def forward(self, x):
        if dist.same_sbp(self.weight.sbp, dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)])):
            # if the last dim of weight sbp sign is S(1), the last dim of x sbp sign must be B.
            if self.weight.sbp[-1] == flow.sbp.split(1):
                x_sbp = x.sbp[:-1] + (flow.sbp.broadcast,)
                x = x.to_consistent(sbp=x_sbp)

            # x.grad sbp must be x.sbp, otherwise backward pass cannot be performed correctly.
            x = x.to_consistent(grad_sbp=x.sbp)
            x = flow.matmul(x, self.weight)

        elif dist.same_sbp(
            self.weight.sbp, dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])
        ):
            # if the last dim of weight sbp sign is S(0), the last dim of x sbp
            # sign must be S(ndim-1).
            if self.weight.sbp[-1] == flow.sbp.split(0):
                x_sbp = x.sbp[:-1] + (flow.sbp.split(x.ndim - 1),)
                x = x.to_consistent(sbp=x_sbp)
                out_sbp = x.sbp[:-1] + (flow.sbp.broadcast,)
            else:
                out_sbp = x.sbp

            x = flow.matmul(x, self.weight)
            # Change x.sbp for followup forward pass.
            # This line can be removed when sbp can be auto inferred.
            x = x.to_consistent(sbp=out_sbp)
        elif dist.same_sbp(
            self.weight.sbp, dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        ):
            # x.grad sbp must be x.sbp, otherwise backward pass cannot be performed correctly.
            x = x.to_consistent(grad_sbp=x.sbp)
            # Change x.sbp to [S(0), S(0)] if weight is [B, B]
            x = x.to_consistent(sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.split(0)]))
            x = flow.matmul(x, self.weight)
        else:
            raise NotImplementedError(f"Not support weight with sbp: {self.weight.sbp}")

        if self.bias is not None:
            if self.skip_bias_add:
                return x, self.bias
            else:
                return x + self.bias
        else:
            return x

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, parallel={}".format(
            self.in_features, self.out_features, self.bias is not None, self.parallel,
        )


# Give an alias for Linear1d
Linear = Linear1D
