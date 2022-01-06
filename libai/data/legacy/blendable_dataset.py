# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Blendable dataset."""

import time
import logging

from collections import OrderedDict
import numpy as np
import oneflow as flow


logger = logging.getLogger(__name__)

class BlendableDataset(flow.utils.data.Dataset):

    def __init__(self, datasets, weights):

        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = 0
        for dataset in self.datasets:
            self.size += len(dataset)

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indecies.
        start_time = time.time()
        assert num_datasets < 255
        self.dataset_index = np.zeros(self.size, dtype=np.uint8)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)

        from libai.data import helpers
        logger.info("building blending indices...")
        helpers.build_blending_indices(self.dataset_index,
                                       self.dataset_sample_index,
                                       weights, num_datasets, self.size,
                                       flow.env.get_rank() == 0)
        logger.info("> elapsed time for building blendable dataset indices: "
                    "{:.2f} (sec)".format(time.time() - start_time))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def supports_prefetch(self):
        return all(d.supports_prefetch for d in self.datasets)

    def prefetch(self, indices):
        group_by_dataset = OrderedDict(list)
        for idx in indices:
            dataset_idx = self.dataset_index[idx]
            sample_idx = self.dataset_sample_index[idx]
            group_by_dataset[dataset_idx].append(sample_idx)
        
        for dataset_idx, group_indice in group_by_dataset:
            self.datasets[dataset_idx].prefetch(group_indice)
