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

logger = logging.getLogger(__name__)


def split_ds(ds, split=None, shuffle=False, save_splits=None, load_splits=None):
    """
    Split a dataset into subsets given proportions of how
    much to allocate per split. If a split is 0% returns None for that split.
    Purpose: Useful for creating train/val/test splits
    Arguments:
        ds (Dataset or array-like): Data to be split.
        split (1D array-like): proportions to split `ds`. `sum(splits) != 0`
    """
    if split is None:
        split = [0.8, 0.2, 0.0]
    split_sum = sum(split)
    if split_sum == 0:
        raise Exception("Split cannot sum to 0.")
    split = np.array(split)
    split /= split_sum
    ds_len = len(ds)
    inds = np.arange(ds_len)
    if shuffle:
        rng = np.random.RandomState(1234)
        rng.shuffle(inds)
    if load_splits is not None:
        inds = np.load(load_splits)
        assert len(inds) == ds_len
        logger.info(f"Load split indices from {load_splits}")
    elif save_splits is not None:
        if flow.env.get_rank() == 0:
            np.save(save_splits, inds)
            logger.info(f"Save split indices to {save_splits}")
    start_idx = 0
    residual_idx = 0
    rtn_ds = [None] * len(split)
    for i, f in enumerate(split):
        if f != 0:
            proportion = ds_len * split[i]
            residual_idx += proportion % 1
            split_ = int(int(proportion) + residual_idx)
            split_inds = inds[start_idx : start_idx + max(split_, 1)]
            rtn_ds[i] = SplitDataset(ds, split_inds)
            start_idx += split_
            residual_idx %= 1
    return rtn_ds


class SplitDataset(flow.utils.data.Dataset):
    """ """

    def __init__(self, dataset, split_inds):
        self.split_inds = list(split_inds)
        self.wrapped_data = dataset

    def __len__(self):
        return len(self.split_inds)

    def __getitem__(self, index):
        return self.wrapped_data[self.split_inds[index]]

    @property
    def supports_prefetch(self):
        return self.wrapped_data.supports_prefetch

    def prefetch(self, indices):
        self.wrapped_data.prefetch(indices)
