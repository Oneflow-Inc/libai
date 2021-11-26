# -*- coding: utf-8 -*-
# Copyright (c) OneFlow, Inc. and its affiliates.

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
    def __init__(self):
        from global_vars import get_args

        args = get_args()
        self._init_parallel_size(args)
        self._init_placement_group(args)
        self._init_parallel_hierarchy()

    def _init_parallel_size(self, args):
        self.world_size_ = args.num_gpus_per_node * args.num_nodes

        # tensor model parallel size.
        self.tmp_size_ = min(args.tensor_model_parallel_size, self.world_size_)
        assert self.world_size_ % self.tmp_size_ == 0, (
            f"world size ({self.world_size_}) is not divisible by"
            f" tensor model parallel size ({self.tmp_size_})"
        )

        ws = self.world_size_ // args.tensor_model_parallel_size
        # pipeline model parallel size.
        self.pmp_size_ = min(args.pipeline_model_parallel_size, ws)

        self.mp_size_ = self.pmp_size_ * self.tmp_size_

        assert self.world_size_ % self.mp_size_ == 0, (
            f"world size ({self.world_size_}) is not divisible by"
            f" tensor model parallel size ({self.tmp_size_}) times"
            f" pipeline model parallel size ({self.pmp_size_})"
        )

        # data parallel world size
        self.dp_size_ = self.world_size_ // self.mp_size_

    def _init_placement_group(self, args):
        node_ids = [i // args.num_gpus_per_node for i in range(self.world_size_)]
        device_ids = list(range(args.num_gpus_per_node)) * args.num_nodes

        # [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
        devices = [(n, d) for n, d in zip(node_ids, device_ids)]
        num_devices_per_stage = self.world_size_ // self.pmp_size_
        stages_devices = [
            _merge_devices(devices[i : (i + num_devices_per_stage)])
            for i in range(0, self.world_size_, num_devices_per_stage)
        ]

        assert args.num_layers % self.pmp_size_ == 0, (
            f"number of layers ({args.num_layers}) is not divisible by"
            f" pipeline model parallel size ({self.pmp_size_})"
        )
        num_layers_per_stage = args.num_layers // self.pmp_size_

        self.layers_stage_ids_ = [
            i // num_layers_per_stage for i in range(args.num_layers)
        ]
        self.layers_devices_ = [
            stages_devices[stage_id] for stage_id in self.layers_stage_ids_
        ]

    def _init_parallel_hierarchy(self):
        if self.is_data_model_parallel():
            self.parallel_hierarchy_ = (
                self.dp_size_,
                self.tmp_size_,
            )
        else:
            self.parallel_hierarchy_ = None

    @property
    def parallel_hierarchy(self):
        return self.parallel_hierarchy_

    @property
    def tensor_model_parallel_size(self):
        return self.tmp_size_

    @property
    def pipeline_model_parallel_size(self):
        return self.pmp_size_

    @property
    def model_parallel_size(self):
        return self.tmp_size_ * self.pmp_size_

    @property
    def data_parallel_size(self):
        return self.dp_size_

    def get_layer_devices(self, layer_idx):
        return self.layers_devices_[layer_idx]

    def get_layer_stage_id(self, layer_idx):
        return self.layers_stage_ids_[layer_idx]

    def is_tensor_model_parallel(self):
        return self.tmp_size_ > 1

    def is_data_parallel(self):
        return self.dp_size_ > 1

    def is_pipeline_model_parallel(self):
        return self.pmp_size_ > 1

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
        dist_util.parallel_hierarchy_,
    )


def get_all_placement(device_type="cuda"):
    dist_util = get_dist_util()

    # FIXME(Lxy): fix this when training with multi-node
    return flow.placement(
        device_type, {0: range(get_world_size())}, dist_util.parallel_hierarchy_
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
    return flow.env.get_rank() // dist_util.mp_size_


def get_data_parallel_world_size():
    dist_util = get_dist_util()
    return dist_util.data_parallel_size


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


def gather(data, dist=0):
    if get_world_size() == 1:
        return [data]

    if not isinstance(data, flow.Tensor):
        tensor = flow.tensor(
            data,
            sbp=get_nd_sbp(flow.sbp.split(0), flow.sbp.split(1)),
            placement=get_all_placement(),
        )
    else:
        tensor = data

    rank = get_rank()

    if rank == dist:
        return ttol(tensor)
    else:
        return []


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
