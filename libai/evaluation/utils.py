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
from collections.abc import Mapping

import oneflow as flow

from libai.utils import distributed as dist


def pad_batch(x_dict, batch_size, last_batch_lack, is_last_batch):
    x = list(x_dict.values())[0]
    tensor_batch = x.shape[0]
    assert tensor_batch <= batch_size

    if tensor_batch == batch_size and not is_last_batch:
        return x_dict, batch_size

    valid_sample = tensor_batch - last_batch_lack
    data_parallel_size = dist.get_data_parallel_size()
    assert tensor_batch % data_parallel_size == 0
    tensor_micro_batch_size = tensor_batch // data_parallel_size
    padded_dict = {}
    for key, xi in x_dict.items():
        pad_shape = (batch_size, *xi.shape[1:])
        local_xi = xi.to_global(
            sbp=flow.sbp.broadcast, placement=flow.env.all_device_placement("cuda")
        ).to_local()
        padded_xi = flow.zeros(pad_shape, dtype=xi.dtype, device="cuda")
        padded_xi[:tensor_batch, ...] = padded_xi[:tensor_batch, ...] + local_xi
        for i in range(last_batch_lack - 1):
            start_idx = tensor_micro_batch_size * (data_parallel_size - i - 1) - 1
            padded_xi[start_idx:-1] = padded_xi[start_idx + 1 :]
        padded_xi = padded_xi.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), placement=xi.placement
        ).to_global(sbp=xi.sbp)
        padded_dict[key] = padded_xi
    return padded_dict, valid_sample


def print_csv_format(results):
    """
    Print main metrics in a particular format
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    assert isinstance(results, Mapping) or not len(results), results
    logger = logging.getLogger(__name__)
    for task, res in results.items():
        if isinstance(res, Mapping):
            # Don't print "AP-category" metrics since they are usually not tracked.
            important_res = [(k, v) for k, v in res.items() if "-" not in k]
            logger.info("copypaste: Task: {}".format(task))
            logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
            logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
        else:
            logger.info(f"copypaste: {task}={res}")


def flatten_results_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r
