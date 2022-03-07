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


import unittest

import numpy as np
import oneflow as flow
import oneflow.unittest
from omegaconf import DictConfig
from oneflow import nn

from libai.layers import Linear
from libai.utils import distributed as dist


class TestLinear(flow.unittest.TestCase):
    def setUp(self):
        # input with shape (8, 8)
        self.inputs = flow.tensor(
            [
                [0.6483, 0.5238, 0.2836, 0.2405, 0.6525, 0.9363, 0.7057, 0.2979],
                [0.6407, 0.7413, 0.1853, 0.6440, 0.3329, 0.0206, 0.0117, 0.6693],
                [0.8074, 0.8030, 0.3937, 0.5001, 0.6239, 0.9128, 0.0208, 0.7877],
                [0.5831, 0.5036, 0.9892, 0.7723, 0.4927, 0.3971, 0.2307, 0.0390],
                [0.8926, 0.8352, 0.6854, 0.2981, 0.7442, 0.7143, 0.7869, 0.3029],
                [0.3492, 0.7022, 0.2281, 0.0468, 0.8628, 0.0579, 0.7481, 0.1385],
                [0.3555, 0.9038, 0.5928, 0.5363, 0.4416, 0.3925, 0.6493, 0.9054],
                [0.0815, 0.3006, 0.0639, 0.9997, 0.3412, 0.2741, 0.0532, 0.7298],
            ],
        )

        # initialize weight with shape (4, 8)
        # this can load into nn.Linear(8, 4)
        self.weight = flow.tensor(
            [
                [0.2143, 0.4756, 0.9593, 0.9044, 0.5790, 0.4611, 0.6538, 0.0862],
                [0.1493, 0.0223, 0.4827, 0.5335, 0.6581, 0.4714, 0.4743, 0.1641],
                [0.3930, 0.3755, 0.1090, 0.8242, 0.8702, 0.2243, 0.1612, 0.7665],
                [0.7100, 0.6721, 0.2446, 0.4741, 0.8203, 0.3415, 0.3371, 0.7093],
            ]
        )
        self.bias = flow.tensor([0.5721, 0.4765, 0.4740, 0.0337])

    @unittest.skipIf(not flow.cuda.is_available(), "only test gpu cases")
    @flow.unittest.skip_unless_1n1d()
    def test_nn_linear(self):
        dist.setup_dist_util(
            DictConfig(
                dict(
                    data_parallel_size=1,
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                )
            )
        )

        # move local tensor to cuda
        inputs = self.inputs.to("cuda")
        weight = self.weight.to("cuda")
        bias = self.bias.to("cuda")

        nn_linear = nn.Linear(8, 4).to("cuda")
        nn_linear.weight.data.copy_(weight)
        nn_linear.bias.data.copy_(bias)
        nn_output = nn_linear(inputs)

        # change local tensor to global tensor
        inputs_g = self.inputs.to_global(
            sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0)
        )
        weight_g = self.weight.to_global(
            sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0)
        )
        bias_g = self.bias.to_global(sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0))

        libai_linear = Linear(8, 4)
        libai_linear.weight.data.copy_(weight_g)
        libai_linear.bias.data.copy_(bias_g)
        libai_output = libai_linear(inputs_g)

        self.assertTrue(np.allclose(nn_output.cpu().numpy(), dist.tton(libai_output), 1e-7, 1e-7))

    @flow.unittest.skip_unless_1n2d()
    def test_col_parallel_linear(self):
        dist.setup_dist_util(
            DictConfig(
                dict(
                    data_parallel_size=1,
                    tensor_parallel_size=2,
                    pipeline_parallel_size=1,
                )
            )
        )

        # move local tensor to cuda
        inputs = self.inputs.to("cuda")
        weight = self.weight.to("cuda")
        bias = self.bias.to("cuda")

        nn_linear = nn.Linear(8, 4).to("cuda")
        nn_linear.weight.data.copy_(weight)
        nn_linear.bias.data.copy_(bias)
        nn_output = nn_linear(inputs)

        # change local tensor to global tensor
        inputs_g = self.inputs.to_global(
            sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0)
        )
        weight_g = self.weight.to_global(
            sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0)
        ).to_global(sbp=flow.sbp.split(0))
        bias_g = self.bias.to_global(
            sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0)
        ).to_global(sbp=flow.sbp.split(0))

        libai_linear = Linear(8, 4, parallel="col")
        libai_linear.weight.data.copy_(weight_g)
        libai_linear.bias.data.copy_(bias_g)
        libai_output = libai_linear(inputs_g)

        self.assertTrue(np.allclose(nn_output.cpu().numpy(), dist.tton(libai_output), 1e-7, 1e-7))

    @flow.unittest.skip_unless_1n2d()
    def test_row_parallel_linear(self):
        dist.setup_dist_util(
            DictConfig(
                dict(
                    data_parallel_size=1,
                    tensor_parallel_size=2,
                    pipeline_parallel_size=1,
                )
            )
        )

        # move local tensor to cuda
        inputs = self.inputs.to("cuda")
        weight = self.weight.to("cuda")
        bias = self.bias.to("cuda")

        nn_linear = nn.Linear(8, 4).to("cuda")
        nn_linear.weight.data.copy_(weight)
        nn_linear.bias.data.copy_(bias)
        nn_output = nn_linear(inputs)

        # change local tensor to global tensor
        inputs_g = self.inputs.to_global(
            sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0)
        ).to_global(sbp=flow.sbp.split(1))
        weight_g = self.weight.to_global(
            sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0)
        ).to_global(sbp=flow.sbp.split(1))
        bias_g = self.bias.to_global(sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0))

        libai_linear = Linear(8, 4, parallel="row")
        libai_linear.weight.data.copy_(weight_g)
        libai_linear.bias.data.copy_(bias_g)
        libai_output = libai_linear(inputs_g)

        self.assertTrue(np.allclose(nn_output.cpu().numpy(), dist.tton(libai_output), 1e-7, 1e-7))
