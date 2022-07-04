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
    r"""Linear layer with 1D parallelism which includes column parallelism and row parallelism.
    The linear layer is defined as :math:`y = xA^T + b`.

    In column parallelism, A^T is parallelized along the second dimension
    as :math:`A^T = [A_1, ..., A_p]`.

    In row parallelism, A^T is parallelized along the first dimension and X along its second
    dimension as:

    .. math::
        A^T = \begin{bmatrix}
                 A\_1 \\
                 . \\
                 . \\
                 . \\
                 A\_p
        \end{bmatrix}
        x = \begin{bmatrix}
                 x\_1 & ... & x\_p
        \end{bmatrix}

    Arguments:
        in_features: size of each input sample.
        out_features: size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias. Defaults to ``True``.
        parallel: Parallel mode. Defaults to "data".
        init_method: method to initialize weight. Defaults to :func:`nn.init.xavier_normal_`.
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
            # Column parallel
            # weight sbp sign: [B, S(0)], weight will be transposed when performing matmul
            # so weight sbp sign actually be [B, S(1)]
            # bias sbp sign: [B, S(0)]
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])
            bias_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])
        elif parallel == "row":
            # Row parallel
            # weight sbp sign: [B, S(1)], weight will be transposed when performing matmul
            # so weight sbp sign actually be [B, S(1)]
            # bias sbp sign: [B, B]
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)])
            bias_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        elif parallel == "data":
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
            bias_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        else:
            raise KeyError(f"{parallel} is not supported! Only support ('data', 'row' and 'col')")

        self.weight = flow.nn.Parameter(
            flow.empty(
                (out_features, in_features),
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
        if dist.same_sbp(self.weight.sbp, dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])):
            # If the last dim of weight sbp sign is S(0), then last dim of weight.t sbp
            # sign is S(1), so the last dim of x sbp sign must be B.
            if self.weight.sbp[-1] == flow.sbp.split(0):
                x_sbp = x.sbp[:-1] + (flow.sbp.broadcast,)
                x = x.to_global(sbp=x_sbp)

            # x.grad sbp must be x.sbp, otherwise backward pass cannot be performed correctly.
            x = x.to_global(grad_sbp=x.sbp)
            x = flow.matmul(x, self.weight, transpose_b=True)

        elif dist.same_sbp(
            self.weight.sbp, dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)])
        ):
            # If the last dim of weight sbp sign is S(1), then last dim of weight.t sbp
            # sign is S(0), so the last dim of x sbp sign must be S(ndim-1).
            if self.weight.sbp[-1] == flow.sbp.split(1):
                x_sbp = x.sbp[:-1] + (flow.sbp.split(x.ndim - 1),)
                x = x.to_global(sbp=x_sbp)
                out_sbp = x.sbp[:-1] + (flow.sbp.broadcast,)
            else:
                out_sbp = x.sbp

            x = flow.matmul(x, self.weight, transpose_b=True)
            # Change x.sbp for followup forward pass.
            # This line can be removed when sbp can be auto inferred.
            x = x.to_global(sbp=out_sbp)
        elif dist.same_sbp(
            self.weight.sbp, dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        ):
            # x.grad sbp must be x.sbp, otherwise backward pass cannot be performed correctly.
            x = x.to_global(grad_sbp=x.sbp)
            # NOTE(chengcheng): when input x is [S(0), B], there is no need to change sbp for x.
            # x = x.to_global(sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.split(0)]))
            x = flow.matmul(x, self.weight, transpose_b=True)
        else:
            # Not supported weight_sbp, deduce sbp and communicate with nccl automatically.
            x = flow.matmul(x, self.weight, transpose_b=True)

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


# Give an alias for Linear1d
Linear = Linear1D
