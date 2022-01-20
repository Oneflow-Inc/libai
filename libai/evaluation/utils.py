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


def pad_batch(x_list, batch_size):
    x = x_list[0]
    valid_sample = x.shape[0]
    assert valid_sample <= batch_size
    # check all batch size is equal
    for xi in x_list[1:]:
        assert xi.shape[0] == valid_sample

    if valid_sample == batch_size:
        return x_list, batch_size
    # pad all data
    padded_list = []
    for xi in x_list:
        pad_shape = (batch_size, *xi.shape[1:])
        padded_xi = flow.zeros(pad_shape, sbp=xi.sbp, placement=xi.placement, dtype=xi.dtype)
        padded_xi[:valid_sample, ...] = padded_xi[:valid_sample, ...] + xi
        padded_xi[valid_sample:, ...] = padded_xi[valid_sample:, ...] + xi[0].unsqueeze(0)
        padded_list.append(padded_xi)
    return padded_list, valid_sample


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron,
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
