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

from omegaconf import OmegaConf
from oneflow.utils.data import DataLoader
from oneflow.utils.data.dataset import ConcatDataset

from libai.config import LazyCall, instantiate
from libai.utils import distributed as dist

from .data_utils import get_train_valid_test_split_
from .samplers import CyclicSampler, SingleRoundSampler
from .structures import Instance


def build_nlp_train_val_test_loader(
    dataset,
    splits,
    weights,
    train_val_test_num_samples,
    train_batch_size,
    test_batch_size,
    train_sampler=LazyCall(CyclicSampler)(shuffle=True),
    test_sampler=LazyCall(SingleRoundSampler)(shuffle=False, drop_last=False),
    num_workers=4,
    consumed_samples=0,
    seed=0,
    collate_fn=None,
    dataset_mixer=ConcatDataset,
):
    """
    Build nlp train_val_test dataloader, used for dataset lack of valid/test dataset

    Returns:
        It will return train/valid/test dataloader

            * train_loader: dataloader for training
            * valid_loader: dataloader for validation
            * test_loader: dataloader for testing

    Arguments:
        dataset: dataset from which to load the data. e.g.: dataset or [dataset1, dataset2, ...]
        splits: ratio config for spliting dataset to train/valid/test. e.g.: [[7, 2, 1], ...]
        weights: ratio config for concate dataset list (Not Supported yet). e.g.: [1.0, ...]
        train_batch_size: how many samples per batch to load in training (micro-batch-size per GPU).
        test_batch_size: how many samples per batch to load in testing (micro-batch-size per GPU).
        sampler:  defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented.
        num_workers: how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``4``).
        consumed_samples: the number of samples that have been trained at the current time,
            used for resuming training (default: ``0``).
        seed: random seed, used for reproducing experiments (default: ``0``).
        collate_fn: merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        dataset_mixer: function for concating list dataset.
    """

    def build_dataset(index, dataset):
        doc_idx_ptr = indexed_dataset.get_doc_idx()
        start_index = ds_splits[index]
        end_index = ds_splits[index + 1] + 1
        indexed_dataset.set_doc_idx(doc_idx_ptr[start_index:end_index])
        dataset.indexed_dataset = indexed_dataset
        dataset.max_num_samples = train_val_test_num_samples[index]
        dataset = instantiate(dataset)

        # Set the original pointer so dataset remains the main dataset.
        indexed_dataset.set_doc_idx(doc_idx_ptr)
        # check
        assert indexed_dataset.doc_idx[0] == 0
        assert indexed_dataset.doc_idx.shape[0] == (total_num_of_documents + 1)
        return dataset

    if OmegaConf.is_list(dataset):
        dataset = list(dataset)
    elif not isinstance(dataset, list):
        dataset = [dataset]

    assert len(dataset) == len(splits), "datasets length must equal splits length"
    assert len(dataset) == len(weights), "datasets length must equal weights length"

    train_datasets, val_datasets, test_datasets = [], [], []
    for dst, split in zip(dataset, splits):
        indexed_dataset = instantiate(dst.indexed_dataset)
        total_num_of_documents = indexed_dataset.doc_idx.shape[0] - 1
        ds_splits = get_train_valid_test_split_(total_num_of_documents, split)

        train_dataset = build_dataset(0, dst)
        val_dataset = build_dataset(1, dst)
        test_dataset = build_dataset(2, dst)

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)

    # [dataset, dataset] -> dataset -> dataloader
    train_dataset = dataset_mixer(train_datasets)
    val_dataset = dataset_mixer(val_datasets)
    test_dataset = dataset_mixer(test_datasets)

    collate_fn = trivial_batch_collator if collate_fn is None else collate_fn

    train_loader, _, _ = build_nlp_train_loader(
        dataset=train_dataset,
        train_batch_size=train_batch_size,
        test_batch_size=None,
        sampler=train_sampler,
        num_workers=num_workers,
        consumed_samples=consumed_samples,
        seed=seed,
        collate_fn=collate_fn,
    )

    valid_loader = build_nlp_test_loader(
        dataset=val_dataset,
        test_batch_size=test_batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        seed=seed,
        collate_fn=collate_fn,
    )

    test_loader = build_nlp_test_loader(
        dataset=test_dataset,
        test_batch_size=test_batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        seed=seed,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader, test_loader


def build_nlp_train_loader(
    dataset,
    train_batch_size,
    test_batch_size=None,
    sampler=LazyCall(CyclicSampler)(shuffle=True),
    num_workers=4,
    consumed_samples=0,
    seed=0,
    collate_fn=None,
    dataset_mixer=ConcatDataset,
    **kwargs
):
    """
    Build nlp train dataloader, it's used for train dataset

    Returns:
        It will return train dataloader, and Nonetype for valid/test dataloader

            * train_loader: dataloader for training
            * None: Nonetype
            * None: Nonetype

    Arguments:
        dataset: dataset from which to load the data. e.g.: dataset or [dataset1, dataset2, ...]
        train_batch_size: how many samples per batch to load in training (micro-batch-size per GPU).
        test_batch_size: no use, set it to None.
        sampler:  defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented.
        num_workers: how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``4``).
        consumed_samples: the number of samples that have been trained at the current time,
            used for resuming training (default: ``0``).
        seed: random seed, used for reproducing experiments (default: ``0``).
        collate_fn: merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        dataset_mixer: function for concating list dataset.
    """
    dataset = instantiate(dataset)
    if OmegaConf.is_list(dataset):
        dataset = list(dataset)
    elif not isinstance(dataset, list):
        dataset = [dataset]

    if len(dataset) > 1:
        dataset = dataset_mixer(dataset)
    else:
        dataset = dataset[0]

    sampler.dataset = dataset
    sampler.micro_batch_size = train_batch_size
    sampler.consumed_samples = consumed_samples
    sampler.data_parallel_rank = dist.get_data_parallel_rank()
    sampler.data_parallel_size = dist.get_data_parallel_size()
    sampler.seed = seed
    sampler = instantiate(sampler)

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        **kwargs,
    )

    return dataloader, None, None


def build_nlp_test_loader(
    dataset,
    test_batch_size,
    sampler=LazyCall(SingleRoundSampler)(shuffle=False, drop_last=False),
    num_workers=4,
    seed=0,
    collate_fn=None,
):
    """
    Build nlp test dataloader, it's used for test dataset

    Returns:
        It will return test dataloader

            * test_loader: dataloader for testing

    Arguments:
        dataset: dataset from which to load the data. e.g.: dataset or [dataset1, dataset2, ...]
        test_batch_size: how many samples per batch to load in testing (micro-batch-size per GPU).
        sampler:  defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented.
        num_workers: how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``4``).
        seed: random seed, used for reproducing experiments (default: ``0``).
        collate_fn: merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
    """
    dataset = instantiate(dataset)
    collate_fn = trivial_batch_collator if collate_fn is None else collate_fn

    sampler.dataset = dataset
    sampler.micro_batch_size = test_batch_size
    sampler.data_parallel_rank = dist.get_data_parallel_rank()
    sampler.data_parallel_size = dist.get_data_parallel_size()
    sampler.seed = seed
    sampler = instantiate(sampler)

    test_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate_fn,
    )
    return test_loader


def build_image_train_loader(
    dataset,
    train_batch_size,
    test_batch_size=None,
    sampler=LazyCall(CyclicSampler)(shuffle=True),
    num_workers=4,
    consumed_samples=0,
    seed=0,
    collate_fn=None,
    dataset_mixer=ConcatDataset,
    mixup_func=None,
    **kwargs
):
    """
    Build image train dataloader, it's used for train dataset

    Returns:
        It will return train dataloader, and Nonetype for valid/test dataloader

            * train_loader: dataloader for training
            * None: Nonetype
            * None: Nonetype

    Arguments:
        dataset: dataset from which to load the data. e.g.: dataset or [dataset1, dataset2, ...]
        train_batch_size: how many samples per batch to load in training (micro-batch-size per GPU).
        test_batch_size: no use, set it to None.
        sampler:  defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented.
        num_workers: how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``4``).
        consumed_samples: the number of samples that have been trained at the current time,
            used for resuming training (default: ``0``).
        seed: random seed, used for reproducing experiments (default: ``0``).
        collate_fn: merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        dataset_mixer: function for concating list dataset.
        mixup_func: function for data argumentation.
    """
    dataset = instantiate(dataset)

    if OmegaConf.is_list(dataset):
        dataset = list(dataset)
    elif not isinstance(dataset, list):
        dataset = [dataset]

    if len(dataset) > 1:
        dataset = dataset_mixer(dataset)
    else:
        dataset = dataset[0]

    sampler.dataset = dataset
    sampler.micro_batch_size = train_batch_size
    sampler.consumed_samples = consumed_samples
    sampler.data_parallel_rank = dist.get_data_parallel_rank()
    sampler.data_parallel_size = dist.get_data_parallel_size()
    sampler.seed = seed
    sampler = instantiate(sampler)

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        **kwargs,
    )
    # Bind up mixup_func to dataloader, and this will be used in Trainer.get_batch
    dataloader.mixup_func = instantiate(mixup_func)

    return dataloader, None, None


def build_image_test_loader(
    dataset,
    test_batch_size,
    sampler=LazyCall(SingleRoundSampler)(shuffle=False, drop_last=False),
    num_workers=4,
    seed=0,
    collate_fn=None,
    **kwargs
):
    """
    Build image test dataloader, used for test dataset

    Returns:
        It will return test dataloader

            * test_loader: dataloader for testing

    Arguments:
        dataset: dataset from which to load the data. e.g.: dataset or [dataset1, dataset2, ...]
        test_batch_size: how many samples per batch to load in testing (micro-batch-size per GPU).
        sampler:  defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented.
        num_workers: how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``4``).
        seed: random seed, used for reproducing experiments (default: ``0``).
        collate_fn: merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
    """
    dataset = instantiate(dataset)

    sampler.dataset = dataset
    sampler.micro_batch_size = test_batch_size
    sampler.data_parallel_rank = dist.get_data_parallel_rank()
    sampler.data_parallel_size = dist.get_data_parallel_size()
    sampler.seed = seed
    sampler = instantiate(sampler)

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        **kwargs,
    )


def trivial_batch_collator(batch):
    assert isinstance(batch[0], Instance), "batch[0] must be `instance` for trivial batch collator"
    batch = Instance.stack(batch)
    return batch
