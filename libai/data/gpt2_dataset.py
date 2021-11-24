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

"""dataset for gpt-2."""

import random
import numpy as np
import oneflow as flow
from bisect import bisect_right
from itertools import accumulate
from libai.utils import print_rank_0

from .dataset_utils import build_index_mappings

class GPT2Dataset(flow.utils.data.Dataset):
    def __init__(self, name, indexed_dataset, tokenizer, data_prefix, documents, num_epochs, max_num_samples, max_seq_length=512, seed=1234):
        self.name = name
        self.indexed_dataset = indexed_dataset

        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        self.doc_idx, self.sample_idx, self.shuffle_ids = build_index_mappings(
            self.name, data_prefix, documents, self.indexed_dataset.sizes,
            num_samples, max_seq_length, seed)

    def __len__(self):
        return self.sample_idx.shape[0] - 1
    
    def __getitem__(self, idx):
        idx = self.shuffle_idx[idx]
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                                    offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(self.indexed_dataset.get(
                self.doc_idx[doc_index_l],
                length=offset_l + 1))
            sample = np.concatenate(sample_list)

        return {'text': np.array(sample, dtype=np.int64)}
    
