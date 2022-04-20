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


# --------------------------------------------------------
# References:
# https://github.com/facebookresearch/deit/blob/0c4b8f60/samplers.py
# https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/samplers/repeat_aug.py
# --------------------------------------------------------


class RASampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on flow.utils.data.DistributedSampler

    Arguments:
        dataset: dataset to be sampled.
        micro_batch_size: batch size for per model instance.
        global_batch_size is micro_batch_size times data_parallel_size.
        shuffle: whether to shuffle the dataset.
        data_parallel_rank: local rank for data parallelism.
        data_parallel_size: the size of data parallelism.
        num_repeats: repeat sampling nums for each sample.
        selected_round: determine the number of samples to select per epoch for each rank
        seed: random seed, used for reproducing experiments (default: ``0``).
    """

    def __init__(
        self,
        dataset,
        micro_batch_size,
        shuffle=True,
        consumed_samples=0,
        data_parallel_rank=0,
        data_parallel_size=1,
        num_repeats=3,
        selected_round=256,
        seed=0,
    ):
        self.data_parallel_size = data_parallel_size
        self.dataset = dataset
        self.rank = data_parallel_rank
        self.num_repeats = num_repeats
        self.micro_batch_size = micro_batch_size
        self.actual_batch_size = self.micro_batch_size * self.data_parallel_size
        
        # samples for each rank: dataset size * repeat nums / rank nums
        self.num_samples = int(
            math.ceil(len(self.dataset) * self.num_repeats / self.data_parallel_size)
        )
        
        # the total samples after repeat sampling
        self.total_size = self.num_samples * self.data_parallel_size

        # the real samples nums for each rank without repeat samples
        if selected_round:
            self.data_size = int(math.floor(len(self.dataset) // selected_round * selected_round / self.data_parallel_size))
        else:
            self.data_size = int(math.ceil(len(self.dataset) / self.data_parallel_size))
        
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
            # produce repeats e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2....]
            indices = [ele for ele in indices for i in range(self.num_repeats)]
            padding_size: int = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
            assert len(indices) == self.total_size

            # subsample: force the repeated samples being put into different rank
            indices = indices[self.rank : self.total_size : self.data_parallel_size]
            assert len(indices) == self.num_samples

            indices = iter(indices[: self.data_size])

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