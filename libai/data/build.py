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


from typing import Optional
import omegaconf

import oneflow as flow
import oneflow.utils.data as data
from oneflow.utils.data.dataset import ConcatDataset

from .structures import Instance


def build_image_train_loader(dataset, batch_size, sampler=None, num_workers=4, collate_fn=None, mix_dataset=ConcatDataset, **kwargs):
    """
    Args:
        dataset: Dataset list or single dataset.
        batch_size: Batch-size for each GPU.
    """
    if isinstance(dataset, omegaconf.listconfig.ListConfig):
        dataset = list(dataset)
    else:
        dataset = [dataset]

    dataset = mix_dataset(dataset)

    collate_fn = trivial_batch_collator if collate_fn is None else collate_fn

    if sampler:
        dataloader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
    else:
        dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
    return dataloader, None, None


def build_image_test_loader(dataset, batch_size, sampler=None, num_workers=4, collate_fn=None, **kwargs):
    return data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=collate_fn, **kwargs)


def trivial_batch_collator(batch):
    assert isinstance(
        batch[0], Instance
    ), "batch[0] must be `instance` for trivial batch collator"
    batch = Instance.stack(batch)

    return batch 