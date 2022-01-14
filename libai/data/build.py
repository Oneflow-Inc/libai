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

import oneflow.utils.data as flowdata
from oneflow.utils.data.dataset import ConcatDataset

from .samplers import CyclicSampler, SingleRoundSampler
from .structures import Instance


def build_image_train_loader(
    dataset,
    batch_size,
    sampler=None,
    num_workers=4,
    collate_fn=None,
    dataset_mixer=ConcatDataset,
    **kwargs
):
    """
    Args:
        dataset: Dataset list or single dataset.
        batch_size: Batch-size for each GPU.
    """
    if isinstance(dataset, omegaconf.listconfig.ListConfig):
        dataset = list(dataset)
    elif not isinstance(dataset, list):
        dataset = [dataset]

    if len(dataset) > 1:
        dataset = dataset_mixer(dataset)
    else:
        dataset = dataset[0]

    if sampler is None:
        # TODO: initilize train sampler
        sampler = CyclicSampler()

    dataloader = flowdata.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        **kwargs
    )

    return dataloader, None, None


def build_image_test_loader(
    dataset, batch_size, sampler=None, num_workers=4, collate_fn=None, **kwargs
):

    if sampler is None:
        # TODO: initilize test_sampler
        sampler = SingleRoundSampler()

    return flowdata.DataLoader(
        dataset,
        batch_size=batch_size,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator
        if collate_fn is None
        else collate_fn ** kwargs,
    )


def trivial_batch_collator(batch):
    assert isinstance(
        batch[0], Instance
    ), "batch[0] must be `instance` for trivial batch collator"
    batch = Instance.stack(batch)

    return batch
