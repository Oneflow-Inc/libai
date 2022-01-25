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

from libai.data.data_utils import BlockIndexedDataset
from libai.data.structures import DistTensorData, Instance


class GPT2Dataset(flow.utils.data.Dataset):
    def __init__(self, tokenizer, data_prefix, indexed_dataset, max_seq_length=512):
        self.dataset = BlockIndexedDataset(
            data_prefix, indexed_dataset, max_seq_length=max_seq_length
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = np.array(self.dataset[idx], dtype=np.long)
        input_ids = flow.tensor(text[:-1], dtype=flow.long)
        labels = flow.tensor(text[1:], dtype=flow.long)
        sample = Instance(
            input_ids=DistTensorData(input_ids), labels=DistTensorData(labels, placement_idx=-1),
        )
        return sample

    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch

    def prefetch(self, indices):
        self.dataset.prefetch(indices)
