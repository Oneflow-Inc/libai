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

import math

import oneflow as flow
from oneflow.utils.data import Sampler


class RASampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on flow.utils.data.DistributedSampler
    Arguments:
        dataset: dataset to be sampled.
        shuffle: whether to shuffle the dataset.
        data_parallel_rank: local rank for data parallelism.
        data_parallel_size: the size of data parallelism.
        seed: random seed, used for reproducing experiments (default: ``0``).
    """

    def __init__(
        self, 
        dataset=None,
        micro_batch_size=32,
        shuffle=True,
        consumed_samples=0,
        data_parallel_rank=0,
        data_parallel_size=1,
        num_repeats=3,
        seed = 0,
    ):
        self.data_parallel_size = data_parallel_size
        self.dataset = dataset
        self.rank = data_parallel_rank
        self.num_repeats = num_repeats
        self.micro_batch_size = micro_batch_size
        self.actual_batch_size = self.micro_batch_size * self.data_parallel_size
        # 重复采样后每个rank的样本数量: 数据集 * 采样次数 / data rank总量
        self.num_samples = int(math.ceil(len(self.dataset) * self.num_repeats / self.data_parallel_size))
        # 重复采样后的总样本量
        self.total_size = self.num_samples * self.data_parallel_size
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        # 每个replica实际样本量，即不重复采样时的每个replica的样本量
        self.data_size = int(math.floor(len(self.dataset) // 256 * 256 / self.data_parallel_size))
        self.shuffle = shuffle
        self.consumed_samples = consumed_samples

        self.seed = seed

    def __iter__(self):
        epoch = self.consumed_samples // self.data_size
        batch = []

        while True:
            # deterministically shuffle based on epoch
            if self.shuffle:
                g = flow.Generator()
                g.manual_seed(self.seed + epoch)
                indices = flow.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = flow.arange(start=0, end=len(self.dataset))

            # add extra samples to make it evenly divisible
            indices = [ele for ele in indices for i in range(self.num_repeats)]
            padding_size: int = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
            assert len(indices) == self.total_size

            # subsample: 使得同一个样本的重复版本进入不同的进程（GPU）
            indices = indices[self.rank:self.total_size:self.data_parallel_size]
            assert len(indices) == self.num_samples

            indices = iter(indices[:self.data_size])

            for idx in indices:
                batch.append(idx)
                if len(batch) == self.micro_batch_size:
                    self.consumed_samples += self.actual_batch_size
                    yield batch
                    batch = []

    def __len__(self):
        return self.data_size

    def set_consumed_samples(self, consumed_samples):
        """you can recover the training iteration by setting `consumed_samples`."""
        self.consumed_samples = consumed_samples 

    def set_epoch(self, epoch):
        """used for restoring training status."""
        self.epoch = epoch