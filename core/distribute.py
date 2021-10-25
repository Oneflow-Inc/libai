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
from core import get_args

_DIST_UTIL = None


def _merge_devices(devices):
    node_devices = dict()
    for node_id, device_id in devices:
        if node_id not in node_devices:
            node_devices[node_id] = []

        node_devices[node_id].append(device_id)

    return node_devices


class _DistributeUtil(object):
    def __init__(self):
        args = get_args()
        self._init_parallel_size(args)
        self._init_placement_group(args)
        self._init_parallel_hierarchy()

    def _init_parallel_size(self, args):
        self.world_size = args.num_gpus_per_node * args.num_nodes

        # tensor model parallel size.
        self.tensor_model_parallel_size = min(args.tensor_model_parallel_size, self.world_size)
        assert self.world_size % self.tensor_model_parallel_size == 0, (
            f"world size ({self.world_size}) is not divisible by"
            f" tensor model parallel size ({self.tensor_model_parallel_size})"
        )

        ws = self.world_size // args.tensor_model_parallel_size
        # pipeline model parallel size.
        self.pipeline_model_parallel_size = min(args.pipeline_model_parallel_size, ws)

        self.model_paralle_size = self.pipeline_model_parallel_size * self.tensor_model_parallel_size

        assert self.world_size % self.model_paralle_size == 0, (
            f"world size ({self.world_size}) is not divisible by"
            f" tensor model parallel size ({self.tensor_model_parallel_size}) times"
            f" pipeline model paralle size ({self.pipeline_model_parallel_size})"
        )

        # data parallel size
        self.data_parallel_size = self.world_size // self.model_paralle_size

    def _init_placement_group(self, args):
        node_ids = [i // args.num_gpus_per_node for i in range(self.world_size)]
        device_ids = list(range(args.num_gpus_per_node)) * args.num_nodes

        devices = [(n, d) for n, d in zip(node_ids, device_ids)]
        num_devices_per_stage = self.world_size // self.pipeline_model_parallel_size
        stages_devices = [
            _merge_devices(devices[i : (i + num_devices_per_stage)])
            for i in range(0, self.world_size, num_devices_per_stage)
        ]

        assert args.num_layers % self.pipeline_model_parallel_size == 0, (
            f"number of layers ({args.num_layers}) is not divisible by"
            f" pipeline model parallel size ({self.pipeline_model_parallel_size})"
        )
        num_layers_per_stage = args.num_layers // self.pipeline_model_parallel_size

        self.layers_stage_ids = [
            i // num_layers_per_stage for i in range(args.num_layers)
        ]
        self.layers_devices = [
            stages_devices[stage_id] for stage_id in self.layers_stage_ids
        ]

    def _init_parallel_hierarchy(self):
        if self.is_data_model_parallel():
            self.parallel_hierarchy = (self.data_parallel_size, self.tensor_model_parallel_size)
        else:
            self.parallel_hierarchy = None

    def get_layer_devices(self, layer_idx):
        return self.layers_devices[layer_idx]

    def get_layer_stage_id(self, layer_idx):
        return self.layers_stage_ids[layer_idx]

    def is_tensor_model_parallel(self):
        return self.tensor_model_parallel_size > 1

    def is_data_parallel(self):
        return self.data_parallel_size > 1

    def is_pipeline_model_parallel(self):
        return self.pipeline_model_parallel_size > 1

    def is_data_model_parallel(self):
        return self.is_tensor_model_parallel() and self.is_data_parallel()


def get_dist_util():
    global _DIST_UTIL
    if _DIST_UTIL is None:
        _DIST_UTIL = _DistributeUtil()
    return _DIST_UTIL


def get_layer_placement(layer_idx, device_type="cuda"):
    dist_util = get_dist_util()
    return flow.placement(
        device_type,
        dist_util.get_layer_devices(layer_idx),
        dist_util.parallel_hierarchy,
    )


def get_nd_sbp(sbp_list):
    assert isinstance(sbp_list, list)
    assert len(sbp_list) == 2
    assert all(isinstance(sbp, flow.sbp.sbp) for sbp in sbp_list)

    dist_util = get_dist_util()
    if dist_util.is_data_model_parallel():
        return sbp_list
    elif dist_util.is_data_parallel():
        return sbp_list[:1]
    elif dist_util.is_tensor_model_parallel():
        return sbp_list[1:]
    else:
        return [flow.sbp.broadcast]


def get_hidden_sbp():
    """ hidden states sbp.
    """
    return get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast])
