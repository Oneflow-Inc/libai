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

from libai.libai import data
import oneflow.utils.data as flowdata
from .structures import Instance
from .samplers import CyclicSampler, SingleRoundSampler
from libai.utils import distributed as dist

def build_nlp_train_val_test_loader(
        datasets, 
        splits, 
        weight, 
        batch_size, 
        num_accumulation_steps=1,
        sampler=None,
        num_workers=4,
        collate_fn=None, 
        blendable_dataset=Blendable_dataset
    ):
    """ 
        Build nlp train_val_test dataloder
    """
    assert len(datasets) == len(splits)
    assert len(datasets) == len(weight)

    if not isinstance(datasets, list):
        datasets = [datasets]

    train_datasets, val_datasets, test_datasets = [], [], []
    for dst, split in zip(datasets, splits):
        train_dataset, val_dataset, test_dataset = split_ds(dst, split)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)

    # [dataset, dataset] -> dataset -> dataloader
    train_dataset = blendable_dataset(train_datasets, weight=weight)
    val_dataset = blendable_dataset(val_datasets, weight=weight)
    test_dataset = blendable_dataset(test_datasets, weight=weight)

    collate_fn = trivial_batch_collator if collate_fn is None else collate_fn    
    if sampler is None:
        train_sampler = CyclicSampler(
            dataset=train_dataset,
            micro_batch_size=batch_size,
            shuffle=True,
            consumed_samples=0,
            data_parallel_rank=dist.get_data_parallel_rank(),
            data_parallel_size=dist.get_data_parallel_size(),
            num_accumulation_steps=num_accumulation_steps,
            seed=0,
        )   
    valid_sampler = SingleRoundSampler(
        dataset=val_dataset,
        micro_batch_size=batch_size,
        shuffle=False,
        data_parallel_rank=dist.get_data_parallel_rank(),
        data_parallel_size=dist.get_data_parallel_size(),
        num_accumulation_steps=1,
        seed=0,
        drop_last=False
    )
    test_sampler = SingleRoundSampler(
        dataset=test_dataset,
        micro_batch_size=batch_size,
        shuffle=False,
        data_parallel_rank=dist.get_data_parallel_rank(),
        data_parallel_size=dist.get_data_parallel_size(),
        num_accumulation_steps=1,
        seed=0,
        drop_last=False
    )

    train_loader = flowdata.DataLoader(
        train_dataset, sampler=train_sampler, num_workers=num_workers, collate_fn=collate_fn)

    evalution_loader = flowdata.DataLoader(
        val_dataset, sampler=valid_sampler, num_workers=num_workers, collate_fn=collate_fn)

    test_loader = flowdata.DataLoader(
        test_dataset, sampler=test_sampler, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, evalution_loader, test_loader


def build_nlp_test_loader(
        dataset, 
        batch_size, 
        sampler=None, 
        num_workers=4, 
        collate_fn=None,
    ):
    """ 
        Build nlp test dataloder
    """
    if sampler is None:
        sampler = SingleRoundSampler(
            dataset=dataset,
            micro_batch_size=batch_size,
            shuffle=False,
            data_parallel_rank=dist.get_data_parallel_rank(),
            data_parallel_size=dist.get_data_parallel_size(),
            num_accumulation_steps=1,
            seed=0,
            drop_last=False
    )
    test_loader = flowdata.DataLoader(
        dataset, sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)
    return test_loader

def trivial_batch_collator(batch):
    assert isinstance(
        batch[0], Instance
    ), "batch[0] must be `instance` for trivial batch collator"
    batch = Instance.stack(batch)
    return batch
