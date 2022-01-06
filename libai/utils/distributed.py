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

_DIST_UTIL = None


def _merge_devices(devices):
    node_devices = dict()
    for node_id, device_id in devices:
        if node_id not in node_devices:
            node_devices[node_id] = []

        node_devices[node_id].append(device_id)

    return node_devices


class _DistributeUtil(object):
    def __init__(self, cfg):
        self._init_parallel_size(cfg)
        self._init_placement_group(cfg)
        self._init_parallel_hierarchy()

    def _init_parallel_size(self, cfg):
        self._world_size = cfg.num_gpus_per_node * cfg.num_nodes

        # tensor parallel size
        self._tensor_parallel_size = min(cfg.tensor_parallel_size, self._world_size)
        assert self._world_size % self._tensor_parallel_size == 0, (
            f"world size ({self._world_size}) is not divisible by"
            f" tensor parallel size ({self._tensor_parallel_size})"
        )
        # Set the actual tensor parallel size to cfg
        cfg.tensor_parallel_size = self._tensor_parallel_size

        # pipeline parallel size
        self._pipeline_parallel_size = min(
            cfg.pipeline_parallel_size, self._world_size // cfg.tensor_parallel_size
        )
        # Set the actual pipeline parallel size to cfg
        cfg.pipeline_parallel_size = self._pipeline_parallel_size

        self._model_parallel_size = (
            self._pipeline_parallel_size * self._tensor_parallel_size
        )

        assert self._world_size % self._model_parallel_size == 0, (
            f"world size ({self._world_size}) is not divisible by"
            f" tensor model parallel size ({self._tensor_parallel_size}) times"
            f" pipeline model parallel size ({self._pipeline_parallel_size})"
        )

        # data parallel size
        self._data_parallel_size = self._world_size // self._model_parallel_size
        # Set the actual data parallel size to cfg
        cfg.data_parallel_size = self._data_parallel_size

    def _init_placement_group(self, cfg):
        node_ids = [i // cfg.num_gpus_per_node for i in range(self._world_size)]
        device_ids = list(range(cfg.num_gpus_per_node)) * cfg.num_nodes

        # [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
        devices = [(n, d) for n, d in zip(node_ids, device_ids)]
        num_devices_per_stage = self._world_size // self._pipeline_parallel_size
        stages_devices = [
            _merge_devices(devices[i : (i + num_devices_per_stage)])
            for i in range(0, self._world_size, num_devices_per_stage)
        ]

        assert cfg.pipeline_num_layers % self._pipeline_parallel_size == 0, (
            f"number of layers ({cfg.pipeline_num_layers}) is not divisible by"
            f" pipeline model parallel size ({self._pipeline_parallel_size})"
        )
        num_layers_per_stage = cfg.pipeline_num_layers // self._pipeline_parallel_size

        self._layers_stage_ids = [
            i // num_layers_per_stage for i in range(cfg.pipeline_num_layers)
        ]
        self._layers_devices = [
            stages_devices[stage_id] for stage_id in self._layers_stage_ids
        ]

    def _init_parallel_hierarchy(self):
        if self.is_data_model_parallel():
            self._parallel_hierarchy = (
                self._data_parallel_size,
                self._tensor_parallel_size,
            )
        else:
            self._parallel_hierarchy = None

    @property
    def parallel_hierarchy(self):
        return self._parallel_hierarchy

    @property
    def tensor_parallel_size(self):
        return self._tensor_parallel_size

    @property
    def pipeline_parallel_size(self):
        return self._pipeline_parallel_size

    @property
    def model_parallel_size(self):
        return self._tensor_parallel_size * self._pipeline_parallel_size

    @property
    def data_parallel_size(self):
        return self._data_parallel_size

    def get_layer_devices(self, layer_idx):
        return self._layers_devices[layer_idx]

    def get_layer_stage_id(self, layer_idx):
        return self._layers_stage_ids[layer_idx]

    def is_tensor_model_parallel(self):
        return self._tensor_parallel_size > 1

    def is_data_parallel(self):
        return self._data_parallel_size > 1

    def is_pipeline_model_parallel(self):
        return self._pipeline_parallel_size > 1

    def is_data_model_parallel(self):
        return self.is_tensor_model_parallel() and self.is_data_parallel()


def setup_dist_util(cfg):
    global _DIST_UTIL
    _DIST_UTIL = _DistributeUtil(cfg)


def get_dist_util():
    global _DIST_UTIL
    assert (
        _DIST_UTIL is not None
    ), "Please setup distributed utils first by invoking `setup_dist_util`!"
    return _DIST_UTIL


def get_layer_placement(layer_idx, device_type="cuda"):
    dist_util = get_dist_util()
    return flow.placement(
        device_type,
        dist_util.get_layer_devices(layer_idx),
        dist_util.parallel_hierarchy,
    )


def get_all_placement(device_type="cuda"):
    dist_util = get_dist_util()

    # FIXME(l1aoxingyu): fix this when training with multi-node
    return flow.placement(
        device_type, {0: range(get_world_size())}, dist_util.parallel_hierarchy
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


def get_data_parallel_rank():
    dist_util = get_dist_util()
    return flow.env.get_rank() // dist_util.model_parallel_size


def get_data_parallel_size():
    dist_util = get_dist_util()
    return dist_util.data_parallel_size


def get_tensor_parallel_size():
    dist_util = get_dist_util()
    return dist_util.tensor_parallel_size


def same_sbp(lhs_sbp, rhs_sbp):
    assert len(lhs_sbp) == len(rhs_sbp)

    for i in range(len(lhs_sbp)):
        if lhs_sbp[i] != rhs_sbp[i]:
            return False
    return True


def get_rank() -> int:
    return flow.env.get_rank()


def get_local_rank() -> int:
    return flow.env.get_local_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def is_last_process() -> bool:
    return get_rank() == get_world_size() - 1


def get_world_size():
    return flow.env.get_world_size()


def ttol(tensor, pure_local=False):
    """ consistent tensor to local tensor"""
    if tensor.is_consistent:
        if pure_local:
            tensor = tensor.to_local()
        else:
            tensor = tensor.to_consistent(
                sbp=get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            ).to_local()

    return tensor


def tton(tensor, local_only=False):
    """ consistent tensor to numpy """
    if tensor.is_consistent:
        tensor = ttol(tensor, local_only)

    return tensor.numpy()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    world_size = get_world_size()
    if world_size == 1:
        return

    flow._oneflow_internal.eager.multi_client.Sync()
