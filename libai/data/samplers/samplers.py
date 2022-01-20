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

import oneflow as flow
from oneflow.utils.data import Sampler


class CyclicSampler(Sampler):
    """This sampler supports cyclic sampling, and it is also compatible with
    non data parallelism and data parallelism.

    Arguments:
        dataset: dataset to be sampled.
        micro_batch_size: batch size for per model instance.
        global_batch_size is micro_batch_size times data_parallel_size.
        shuffle: whether to shuffle the dataset.
        consumed_samples: the number of samples that have been trained at the current time,
        used for resuming training.
        data_parallel_rank: local rank for data parallelism.
        data_parallel_size: the size of data parallelism.
        seed: random seed, used for reproducing experiments.
    """

    def __init__(
        self,
        dataset,
        micro_batch_size,
        shuffle=False,
        consumed_samples=0,
        data_parallel_rank=0,
        data_parallel_size=1,
        seed=0,
    ):
        self.dataset = dataset
        self.data_size = len(self.dataset)
        self.shuffle = shuffle

        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_size = micro_batch_size
        self.actual_batch_size = self.micro_batch_size * self.data_parallel_size
        self.remain_data_size = self.data_size % self.actual_batch_size
        self.active_data_size = self.data_size - self.remain_data_size
        self.consumed_samples = consumed_samples

        self.seed = seed

    def __iter__(self):
        """divide the data into data_parallel_size buckets,
        and shuffle it if `shuffle` is set to `True`.
        Each processor samples from its own buckets and data_loader
        will load the corresponding data.
        """
        epoch = self.consumed_samples // self.data_size
        current_epoch_samples = self.consumed_samples % self.data_size
        batch = []

        while True:
            bucket_size = self.data_size // self.actual_batch_size * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            if self.shuffle:
                generator = flow.Generator()
                generator.manual_seed(self.seed + epoch)
                random_idx = flow.randperm(bucket_size, generator=generator).tolist()
                indices = [start_idx + x for x in random_idx[bucket_offset:]]
            else:
                seq_idx = flow.arange(bucket_size).tolist()
                indices = [start_idx + x for x in seq_idx[bucket_offset:]]

            epoch += 1

            if hasattr(self.dataset, "supports_prefetch") and self.dataset.supports_prefetch:
                self.dataset.prefetch(indices)

            for idx in indices:
                batch.append(idx)
                if len(batch) == self.micro_batch_size:
                    self.consumed_samples += self.actual_batch_size
                    yield batch
                    batch = []

            current_epoch_samples = 0

    def __len__(self):
        return self.data_size

    def set_consumed_samples(self, consumed_samples):
        """you can recover the training iteration by setting `consumed_samples`."""
        self.consumed_samples = consumed_samples

    def set_epoch(self, epoch):
        """used for restoring training status."""
        self.epoch = epoch


class SingleRoundSampler(Sampler):
    """This sampler supports single round sampling, and it is also compatible with
    non data parallelism and data parallelism.

    Arguments:
        dataset: dataset to be sampled.
        micro_batch_size: batch size for per model instance.
        global_batch_size is micro_batch_size times data_parallel_size.
        shuffle: whether to shuffle the dataset.
        data_parallel_rank: local rank for data parallelism.
        data_parallel_size: the size of data parallelism.
        seed: random seed, used for reproducing experiments.
        drop_last: whether to drop the remaining data. Default to `False`.
    """

    def __init__(
        self,
        dataset,
        micro_batch_size,
        shuffle=False,
        data_parallel_rank=0,
        data_parallel_size=1,
        seed=0,
        drop_last=False,
    ):
        self.dataset = dataset
        self.data_size = len(self.dataset)
        self.shuffle = shuffle

        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_size = micro_batch_size

        self.seed = seed
        self.drop_last = drop_last

    def __iter__(self):
        bucket_size = self.data_size // self.data_parallel_size
        remain = self.data_size % self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        if self.data_parallel_rank < remain:
            bucket_size += 1
        start_idx += min(self.data_parallel_rank, remain)

        if self.shuffle:
            generator = flow.Generator()
            generator.manual_seed(self.seed)
            random_idx = flow.randperm(bucket_size, generator=generator).tolist()
            indices = [start_idx + x for x in random_idx]
        else:
            seq_idx = flow.arange(bucket_size).tolist()
            indices = [start_idx + x for x in seq_idx]

        if hasattr(self.dataset, "supports_prefetch") and self.dataset.supports_prefetch:
            self.dataset.prefetch(indices)

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return self.data_size
