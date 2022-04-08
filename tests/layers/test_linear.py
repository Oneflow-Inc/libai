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

        inputs = flow.rand(8, 8, sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0))
        weight = flow.rand(4, 8, sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0))
        bias = flow.rand(4, sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0))

        nn_linear = nn.Linear(8, 4).to("cuda")
        nn_linear.weight.data.copy_(dist.ttol(weight).to("cuda"))
        nn_linear.bias.data.copy_(dist.ttol(bias).to("cuda"))
        nn_output = nn_linear(dist.ttol(inputs).to("cuda"))

        libai_linear = Linear(8, 4)
        libai_linear.weight.data.copy_(weight)
        libai_linear.bias.data.copy_(bias)
        libai_output = libai_linear(inputs)

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

        inputs = flow.rand(8, 8, sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0))
        weight = flow.rand(4, 8, sbp=flow.sbp.split(0), placement=dist.get_layer_placement(0))
        bias = flow.rand(4, sbp=flow.sbp.split(0), placement=dist.get_layer_placement(0))

        nn_linear = nn.Linear(8, 4).to("cuda")
        nn_linear.weight.data.copy_(dist.ttol(weight).to("cuda"))
        nn_linear.bias.data.copy_(dist.ttol(bias).to("cuda"))
        nn_output = nn_linear(dist.ttol(inputs).to("cuda"))

        libai_linear = Linear(8, 4, parallel="col")
        libai_linear.weight.data.copy_(weight)
        libai_linear.bias.data.copy_(bias)
        libai_output = libai_linear(inputs)

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

        inputs = flow.rand(8, 8, sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0))
        weight = flow.rand(4, 8, sbp=flow.sbp.split(1), placement=dist.get_layer_placement(0))
        bias = flow.rand(4, sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0))

        # move local tensor to cuda
        nn_linear = nn.Linear(8, 4).to("cuda")
        nn_linear.weight.data.copy_(dist.ttol(weight).to("cuda"))
        nn_linear.bias.data.copy_(dist.ttol(bias).to("cuda"))
        nn_output = nn_linear(dist.ttol(inputs).to("cuda"))

        libai_linear = Linear(8, 4, parallel="row")
        libai_linear.weight.data.copy_(weight)
        libai_linear.bias.data.copy_(bias)
        libai_output = libai_linear(inputs)

        self.assertTrue(np.allclose(nn_output.cpu().numpy(), dist.tton(libai_output), 1e-7, 1e-7))
