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
import time
from collections import OrderedDict

import numpy as np
import oneflow as flow
from oneflow.utils.data import Sampler

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
        helpers.build_blending_indices(
            self.dataset_index,
            self.dataset_sample_index,
            weights,
            num_datasets,
            self.size,
            flow.env.get_rank() == 0,
        )
        logger.info(
            "> elapsed time for building blendable dataset indices: "
            "{:.2f} (sec)".format(time.time() - start_time)
        )

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
    """
    """

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


class CyclicSampler(Sampler):
    """ This sampler supports cyclic sampling, and it is also compatible with non data
    parallelism and data parallelism.

    Arguments:
        dataset: dataset to be sampled.
        micro_batch_size: batch size for per model instance. global_batch_size is
        micro_batch_size times data_parallel_size.
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
        """ divide the data into data_parallel_size buckets, and shuffle
        it if `shuffle` is set to `True`.
        Each processor samples from its own buckets and data_loader
        will load the corresponding data.
        """
        epoch = self.consumed_samples // self.data_size
        batch = []
        while True:
            current_epoch_samples = self.consumed_samples % self.data_size

            bucket_size = (
                self.data_size // self.actual_batch_size * self.micro_batch_size
            )
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

            if (
                hasattr(self.dataset, "supports_prefetch")
                and self.dataset.supports_prefetch
            ):
                self.dataset.prefetch(indices)

            for idx in indices:
                batch.append(idx)
                if len(batch) == self.micro_batch_size:
                    self.consumed_samples += self.actual_batch_size
                    yield batch
                    batch = []

    def __len__(self):
        return self.data_size

    def set_consumed_samples(self, consumed_samples):
        """you can recover the training iteration by setting `consumed_samplers`."""
        self.consumed_samples = consumed_samples

    def set_epoch(self, epoch):
        """used for restoring training status."""
        self.epoch = epoch


class SingleRoundSampler(Sampler):
    """ This sampler supports single round sampling, and it is also compatible with non
    data parallelism and data parallelism.

    Arguments:
        dataset: dataset to be sampled.
        micro_batch_size: batch size for per model instance. global_batch_size is
        micro_batch_size times data_parallel_size.
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

        if (
            hasattr(self.dataset, "supports_prefetch")
            and self.dataset.supports_prefetch
        ):
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
