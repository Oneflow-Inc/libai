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

import numpy as np
import oneflow as flow

class PretrainingSampler:
    def __init__(self, dataset, micro_batch_size, num_epochs=1, shuffle=False, data_parallel_rank=0, data_parallel_size=1, seed=1234, drop_last=True):
        self.dataset = dataset
        self.data_size = len(self.dataset)
        self.epoch = 0
        self.num_epochs = num_epochs
        self.shuffle = shuffle

        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_size = micro_batch_size
        self.actual_batch_size = self.micro_batch_size * self.data_parallel_size

        self.seed = seed
        self.drop_last = drop_last

    def __iter__(self):
        consumed_samples = 0
        batch = []
        for epoch in range(self.epoch, self.num_epochs):
            current_samples = consumed_samples % self.data_size
            bucket_size = self.data_size // self.actual_batch_size * self.micro_batch_size
            bucket_offset = current_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            if self.shuffle:
                generator = flow.Generator()
                generator.manual_seed(self.seed + epoch)
                random_idx = flow.randperm(bucket_size, generator=generator).tolist()
                indices = [start_idx + x for x in random_idx[bucket_offset:]]
            else:
                seq_idx = flow.arange(bucket_size).tolist()
                indices = [start_idx + x for x in seq_idx[bucket_offset:]]

            for idx in idx_range:
                batch.append(idx)
                if len(batch) == self.micro_batch_size:
                    consumed_samples += self.actual_batch_size
                    yield batch
                    batch = []
        
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return self.data_size * self.num_epochs

    def set_epoch(self, epoch):
        self.epoch = epoch
