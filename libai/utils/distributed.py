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

import logging

import numpy as np
import oneflow as flow
from omegaconf import OmegaConf

from libai.config import try_get_key

logger = logging.getLogger(__name__)

_DIST_UTIL = None


def _merge_devices(devices):
    num_gpus_per_node = get_world_size() // get_num_nodes()
    node_devices = [node_id * num_gpus_per_node + device_id for node_id, device_id in devices]
    return node_devices


class _DistributeUtil(object):
    def __init__(self, cfg):

        self._init_distributed_env(cfg)
        self._init_parallel_size(cfg)
        self._init_placement_group(cfg)
        self._init_parallel_hierarchy()

    def _init_distributed_env(self, cfg):
        """Initialize the distributed environment."""

        num_nodes = get_num_nodes()
        num_gpus_per_node = get_world_size() // num_nodes

        if try_get_key(cfg, "num_gpus_per_node", default=num_gpus_per_node) != num_gpus_per_node:
            # This means key(num_gpus_per_node) saved in config is not equal
            # to environment variable.
            # Give user a warning about inconsistent reproduce environment.
            logger.warning(
                "'train.dist.num_gpus_per_node' are not equal to environment variable. "
                f"{cfg.num_gpus_per_node} != {num_gpus_per_node}"
            )

        if try_get_key(cfg, "num_nodes", default=num_nodes) != num_nodes:
            logger.warning(
                "'train.dist.num_nodes' are not equal to"
                f"environment variable. {cfg.num_nodes} != {num_nodes}"
            )

        # Set the actual value to config
        cfg.num_nodes = num_nodes
        cfg.num_gpus_per_node = num_gpus_per_node

        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self._world_size = num_gpus_per_node * num_nodes

    def _init_parallel_size(self, cfg):

        # tensor parallel size
        self._tensor_parallel_size = min(cfg.tensor_parallel_size, self.world_size)
        assert self.world_size % self._tensor_parallel_size == 0, (
            f"world size ({self.world_size}) is not divisible by"
            f" tensor parallel size ({self._tensor_parallel_size})"
        )
        # Set the actual tensor parallel size to cfg
        cfg.tensor_parallel_size = self._tensor_parallel_size

        # pipeline parallel size
        self._pipeline_parallel_size = min(
            cfg.pipeline_parallel_size, self.world_size // cfg.tensor_parallel_size
        )
        # Set the actual pipeline parallel size to cfg
        cfg.pipeline_parallel_size = self._pipeline_parallel_size

        if cfg.pipeline_parallel_size > 1:
            assert (
                try_get_key(cfg, "pipeline_num_layers") is not None
            ), "cfg.train.dist.pipeline_num_layers must be set when run pipeline parallel"

            assert cfg.pipeline_num_layers >= self._pipeline_parallel_size, (
                f"number of layers ({cfg.pipeline_num_layers}) is less than"
                f" pipeline model parallel size ({self._pipeline_parallel_size})"
            )
            if try_get_key(cfg, "custom_pipeline_stage_id") is not None:
                assert OmegaConf.is_list(
                    cfg.custom_pipeline_stage_id
                ), "type of cfg.train.dist.custom_pipeline_stage_id must be list"
                cfg.custom_pipeline_stage_id = list(cfg.custom_pipeline_stage_id)
                assert max(cfg.custom_pipeline_stage_id) < self._world_size, (
                    f"the element {max(cfg.custom_pipeline_stage_id)} in"
                    " cfg.train.dist.custom_pipeline_stage_id is out of range"
                    f" for total rank {self._world_size}"
                )
                assert len(cfg.custom_pipeline_stage_id) == cfg.pipeline_num_layers, (
                    "the length of cfg.train.dist.custom_pipeline_stage_id"
                    f" {len(cfg.custom_pipeline_stage_id)} must be equal to"
                    " cfg.train.dist.pipeline_num_layers"
                    f" {cfg.train.dist.pipeline_num_layers}"
                )
        else:
            # no pipeline parallel, just set 10000
            if try_get_key(cfg, "pipeline_num_layers") is None:
                cfg.pipeline_num_layers = 10000

        self._model_parallel_size = self._pipeline_parallel_size * self._tensor_parallel_size

        assert self.world_size % self._model_parallel_size == 0, (
            f"world size ({self.world_size}) is not divisible by"
            f" tensor model parallel size ({self._tensor_parallel_size}) times"
            f" pipeline model parallel size ({self._pipeline_parallel_size})"
        )

        # data parallel size
        self._data_parallel_size = self.world_size // self._model_parallel_size
        # Set the actual data parallel size to cfg
        cfg.data_parallel_size = self._data_parallel_size

    def _init_placement_group(self, cfg):
        node_ids = [i // self.num_gpus_per_node for i in range(self.world_size)]
        device_ids = list(range(self.num_gpus_per_node)) * self.num_nodes

        # [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
        devices = [(n, d) for n, d in zip(node_ids, device_ids)]
        num_devices_per_stage = self.world_size // self._pipeline_parallel_size
        stages_devices = [
            _merge_devices(devices[i : (i + num_devices_per_stage)])
            for i in range(0, self.world_size, num_devices_per_stage)
        ]

        # change pipeline_num_layers to make the middle stages contain more layers
        if (
            self._pipeline_parallel_size >= 4
            and cfg.pipeline_num_layers >= 8
            and cfg.pipeline_num_layers % self._pipeline_parallel_size == 0
        ):
            temp_num_layers_per_stage = cfg.pipeline_num_layers // self._pipeline_parallel_size
            actual_pipeline_num_layers = cfg.pipeline_num_layers + min(
                self._pipeline_parallel_size - 1, temp_num_layers_per_stage
            )
        else:
            actual_pipeline_num_layers = cfg.pipeline_num_layers

        num_layers_per_stage = actual_pipeline_num_layers // self._pipeline_parallel_size
        stage_offset = actual_pipeline_num_layers % self._pipeline_parallel_size

        # stage_offset can make the later stages contain more layers when pipeline_num_layers
        # cannot be divided by pipeline_parallel_size.
        # This can make pipeline parallel more memory efficient.
        self._layer_stage_ids = []
        for i in range(0, actual_pipeline_num_layers - stage_offset, num_layers_per_stage):
            stage_id = i // num_layers_per_stage
            if stage_id >= (self._pipeline_parallel_size - stage_offset):
                self._layer_stage_ids.append(stage_id)
            self._layer_stage_ids.extend([stage_id] * num_layers_per_stage)
        self._layer_stage_ids = self._layer_stage_ids[: cfg.pipeline_num_layers]
        # when pipeline_parallel_size > 1, we add pipeline_stage_id infomation into cfg
        if cfg.pipeline_parallel_size > 1:
            cfg.auto_pipeline_stage_id = self._layer_stage_ids
            # set pipeline_stage_id by users' setting
            if try_get_key(cfg, "custom_pipeline_stage_id") is not None:
                self._layer_stage_ids = cfg.custom_pipeline_stage_id
            cfg.actual_pipeline_stage_id = self._layer_stage_ids

        self._layer_ranks = [stages_devices[stage_id] for stage_id in self._layer_stage_ids]

    def _init_parallel_hierarchy(self):
        if self.is_data_model_parallel():
            self._parallel_hierarchy = (
                self._data_parallel_size,
                self._tensor_parallel_size,
            )
        else:
            self._parallel_hierarchy = None

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_gpus_per_node(self):
        return self._num_gpus_per_node

    @property
    def world_size(self):
        return self._world_size

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
        return self._tensor_parallel_size

    @property
    def data_parallel_size(self):
        return self._data_parallel_size

    def get_layer_ranks(self, layer_idx):
        layer_ranks = self._layer_ranks[layer_idx]
        if self._parallel_hierarchy is None:
            return layer_ranks
        else:
            assert len(self._parallel_hierarchy) == 2
            return np.asarray(layer_ranks).reshape(self._parallel_hierarchy).tolist()

    def get_layer_stage_id(self, layer_idx):
        return self._layer_stage_ids[layer_idx]

    def is_tensor_model_parallel(self):
        return self._tensor_parallel_size > 1

    def is_data_parallel(self):
        return self._data_parallel_size > 1

    def is_pipeline_model_parallel(self):
        return self._pipeline_parallel_size > 1

    def is_data_model_parallel(self):
        return self.is_tensor_model_parallel() and self.is_data_parallel()


def setup_dist_util(cfg):
    """Initialize the distributed environment with configuration.

    Example:

    .. code-block:: python

        from omegaconf import DictConfig

        # set the hybrid parallel distributed environment with 2D mesh GPUs
        setup_dist_util(
            DictConfig(
                dict(
                    data_parallel_size=2,
                    tensor_parallel_size=2,
                    pipeline_parallel_size=1,
                )
            )
        )

    """
    global _DIST_UTIL
    _DIST_UTIL = _DistributeUtil(cfg)


def get_dist_util():
    """Get distributed utils if it's been setup. Otherwise, initialize it with
    single node/single gpu environment."""
    global _DIST_UTIL
    if _DIST_UTIL is None:
        logger.warning(
            "Distributed env is not set up, configure it by default (single node, single gpu)."
        )
        from omegaconf import DictConfig

        setup_dist_util(
            DictConfig(
                dict(
                    data_parallel_size=1,
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                )
            )
        )
    return _DIST_UTIL


def get_layer_placement(layer_idx, device_type="cuda"):
    """
    Get ``flow.placement`` object with the initialized distributed environment
    according to the ``layer_idx``.

    Args:
        layer_idx (int): layer index indicating the rank groups. This is very useful for pipeline
            parallelism training where different layers are on different ranks.
        device_type (str, optional): device type. Defaults to "cuda".
    """
    dist_util = get_dist_util()
    if not flow.cuda.is_available() and device_type == "cuda":
        device_type = "cpu"
    return flow.placement(
        device_type,
        dist_util.get_layer_ranks(layer_idx),
    )


def get_nd_sbp(sbp_list):
    """Get nd sbp signature list, which is consistent with 1D/2D mesh GPUs.

    Args:
        sbp_list (list): a sbp list with 2D mesh.

    Returns:
        A modified sbp list according to the initialized distributed environment.
    """
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
    """Hidden states sbp."""
    return get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast])


def get_data_parallel_rank():
    dist_util = get_dist_util()
    return (flow.env.get_rank() // dist_util.model_parallel_size) % dist_util.data_parallel_size


def get_data_parallel_size():
    dist_util = get_dist_util()
    return dist_util.data_parallel_size


def get_tensor_parallel_size():
    dist_util = get_dist_util()
    return dist_util.tensor_parallel_size


def same_sbp(lhs_sbp, rhs_sbp):
    """Determine if two sbp signatures are the same."""
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


def get_num_nodes():
    return flow.env.get_node_size()


def convert_to_distributed_default_setting(module):
    """
    Helper function to convert all eager local tensor in :attr:`nn.Module` in the model to
    global tensor with data parallelism as default.
    """
    for _, v in module.state_dict().items():
        if not v.is_global:
            module.to_global(
                sbp=get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=get_layer_placement(0),
            )
            return


def ttol(tensor, pure_local=False, ranks=None):
    """Global tensor to local tensor."""
    if tensor.is_global:
        placement = tensor.placement if not ranks else flow.placement("cuda", ranks)
        if pure_local:
            tensor = tensor.to_global(placement=placement).to_local()
        else:
            tensor = tensor.to_global(
                sbp=get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), placement=placement
            ).to_local()

    return tensor


def tton(tensor, local_only=False, ranks=None):
    """Global tensor to numpy ndarray."""
    if tensor.is_global:
        tensor = ttol(tensor, local_only, ranks)

    return tensor.numpy()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training.
    """
    world_size = get_world_size()
    if world_size == 1:
        return

    flow.comm.barrier()
