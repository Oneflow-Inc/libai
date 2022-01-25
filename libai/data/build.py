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


import omegaconf
from oneflow.utils.data import DataLoader
from oneflow.utils.data.dataset import ConcatDataset

from libai.utils import distributed as dist

from .data_utils import split_ds
from .samplers import CyclicSampler, SingleRoundSampler
from .structures import Instance


def build_nlp_train_val_test_loader(
    dataset,
    splits,
    weights,
    train_batch_size,
    test_batch_size,
    sampler=None,
    num_workers=4,
    consumed_samples=0,
    seed=0,
    collate_fn=None,
    dataset_mixer=ConcatDataset,
):
    """
    Build nlp train_val_test dataloader
    """
    # TODO: add input type, add dataset_weights sampler
    if isinstance(dataset, omegaconf.listconfig.ListConfig):
        dataset = list(dataset)
    elif not isinstance(dataset, list):
        dataset = [dataset]

    assert len(dataset) == len(splits), "datasets length must equal splits length"
    assert len(dataset) == len(weights), "datasets length must equal weights length"

    train_datasets, val_datasets, test_datasets = [], [], []
    for dst, split in zip(dataset, splits):
        train_dataset, val_dataset, test_dataset = split_ds(dst, split)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)

    # [dataset, dataset] -> dataset -> dataloader
    train_dataset = dataset_mixer(train_datasets)
    val_dataset = dataset_mixer(val_datasets)
    test_dataset = dataset_mixer(test_datasets)

    collate_fn = trivial_batch_collator if collate_fn is None else collate_fn
    if sampler is None:
        train_sampler = CyclicSampler(
            dataset=train_dataset,
            micro_batch_size=train_batch_size,
            shuffle=True,
            consumed_samples=consumed_samples,
            data_parallel_rank=dist.get_data_parallel_rank(),
            data_parallel_size=dist.get_data_parallel_size(),
            seed=seed,
        )
    valid_sampler = SingleRoundSampler(
        dataset=val_dataset,
        micro_batch_size=test_batch_size,
        shuffle=False,
        data_parallel_rank=dist.get_data_parallel_rank(),
        data_parallel_size=dist.get_data_parallel_size(),
        seed=seed,
        drop_last=False,
    )
    test_sampler = SingleRoundSampler(
        dataset=test_dataset,
        micro_batch_size=test_batch_size,
        shuffle=False,
        data_parallel_rank=dist.get_data_parallel_rank(),
        data_parallel_size=dist.get_data_parallel_size(),
        seed=seed,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=num_workers, collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        val_dataset, batch_sampler=valid_sampler, num_workers=num_workers, collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset, batch_sampler=test_sampler, num_workers=num_workers, collate_fn=collate_fn,
    )

    return train_loader, valid_loader, test_loader


def build_nlp_test_loader(
    dataset, test_batch_size, sampler=None, num_workers=4, seed=0, collate_fn=None,
):
    """
    Build nlp test dataloader
    """
    # TODO: add input type
    collate_fn = trivial_batch_collator if collate_fn is None else collate_fn
    if sampler is None:
        sampler = SingleRoundSampler(
            dataset=dataset,
            micro_batch_size=test_batch_size,
            shuffle=False,
            data_parallel_rank=dist.get_data_parallel_rank(),
            data_parallel_size=dist.get_data_parallel_size(),
            seed=seed,
            drop_last=False,
        )
    test_loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=collate_fn
    )
    return test_loader


def build_image_train_loader(
    dataset,
    train_batch_size,
    test_batch_size=None,
    sampler=None,
    num_workers=4,
    consumed_samples=0,
    seed=42,
    collate_fn=None,
    dataset_mixer=ConcatDataset,
    **kwargs
):
    """
    Args:
        dataset: Dataset list or single dataset.
        batch_size: Batch-size for each GPU.
    """
    # TODO: add input type
    if isinstance(dataset, omegaconf.listconfig.ListConfig):
        dataset = list(dataset)
    elif not isinstance(dataset, list):
        dataset = [dataset]

    if len(dataset) > 1:
        dataset = dataset_mixer(dataset)
    else:
        dataset = dataset[0]

    if sampler is None:
        sampler = CyclicSampler(
            dataset=dataset,
            micro_batch_size=train_batch_size,
            shuffle=True,
            consumed_samples=consumed_samples,
            data_parallel_rank=dist.get_data_parallel_rank(),
            data_parallel_size=dist.get_data_parallel_size(),
            seed=seed,
        )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        **kwargs,
    )

    return dataloader, None, None


def build_image_test_loader(
    dataset, test_batch_size, sampler=None, num_workers=4, seed=42, collate_fn=None, **kwargs
):

    if sampler is None:
        sampler = SingleRoundSampler(
            dataset=dataset,
            micro_batch_size=test_batch_size,
            shuffle=False,
            data_parallel_rank=dist.get_data_parallel_rank(),
            data_parallel_size=dist.get_data_parallel_size(),
            seed=seed,
            drop_last=False,
        )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        **kwargs,
    )


def trivial_batch_collator(batch):
    assert isinstance(batch[0], Instance), "batch[0] must be `instance` for trivial batch collator"
    batch = Instance.stack(batch)
    return batch
