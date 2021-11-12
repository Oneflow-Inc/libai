# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""Split dataset."""

import numpy as np
import oneflow as flow
from core.utils import print_rank_0
from operator import itemgetter

def split_ds(ds, split=[0.8, 0.1, 0.1], shuffle=True):
    split_sum = sum(split)
    assert split_sum != 0, "Split cannot sum to 0."
    split = np.array(split)
    split /= split_sum
    ds_len = len(ds)
    inds = np.arange(ds_len)
    if shuffle:
        np.random.shuffle(inds)
    start_idx = 0
    residual_idx = 0
    rtn_ds = [None] * len(split)
    for i, f in enumerate(split):
        if f != 0:
            proportion = ds_len * split[i]
            residual_idx += proportion % 1
            split_ = int(int(proportion) + residual_idx)
            split_inds = inds[start_idx: start_idx + max(split_, 1)]
            rtn_ds[i] = SplitDataset(ds, split_inds)
            start_idx += split_
            residual_idx %= 1
    return rtn_ds

class SplitDataset(flow.utils.data.Dataset):

    def __init__(self, dataset, split_inds):
        super().__init__()
        self.split_inds = list(split_inds)
        self.wrapped_dataset = dataset
        self.is_lazy = isinstance(ds, lazy_loader) or (hasattr(ds, 'is_lazy') and ds.is_lazy)
        if self.is_lazy:
            self.lens = itemgetter(*self.split_inds)(list(self.wrapped_dataset.lens))
        self._X = None
        self._Y = None

    def __len__(self):
        return len(self.split_inds)

    def __getitem__(self, idx):
        return self.wrapped_dataset[self.split_inds[idx]]

    def SetTokenizer(self, tokenizer):
        self.wrapped_dataset.SetTokenizer(tokenizer)
    
    def GetTokenizer(self):
        return self.wrapped_dataset.GetTokenizer()

    @property
    def X(self):
        if self._X is None:
            self._X = itemgetter(*self.split_inds)(self.wrapped_dataset.X)
        return self._X
    
    @property
    def Y(self):
        if self._Y is None:
            self._Y = np.array(itemgetter(*self.split_inds)(self.wrapped_dataset.Y))
        return self._Y

    def __iter__(self):
        for idx in self.split_inds:
            yield self.wrapped_dataset[idx]

