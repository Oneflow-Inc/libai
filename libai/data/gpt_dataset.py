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

"""dataset for gpt."""

import numpy as np
import oneflow as flow

from .legacy.reindexed_dataset import BlockIndexedDataset


class GPT2Dataset(flow.utils.data.Dataset):
    """"""
    # 这里仍然选择传入num_epochs和max_num_samples参数，是考虑到，如果每个epoch重新循环一遍dataset，可能会出现drop last情形。
    # 传入num_epochs和max_num_samples参数，避免drop last，可以充分使用数据。
    def __init__(self, tokenizer, data_prefix, indexed_dataset, max_seq_length=512):
        self.dataset = BlockIndexedDataset(data_prefix, indexed_dataset, max_seq_length=max_seq_length)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {'text': np.array(sample, dtype=np.int64)}
    
    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch

    def prefetch(self, indices):
        self.dataset.prefetch(indices)
    