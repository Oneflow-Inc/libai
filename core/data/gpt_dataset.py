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
from core import distribute as dist
from core.data import register_dataset

from .base_dataset import BaseDataLoader

@register_dataset("gpt_dataset")
class GPTDataLoader(BaseDataLoader):

    @classmethod
    def build_dataset(cls, args, subset):
        subsets = {'train': 0, 'valid': 1, 'eval': 1, 'test': 2}
        data_loader = GPTDataLoader(
            dataset=args.dataset,
            num_samples=args.num_samples,
            batch_size=args.global_batch_size,
            max_seq_length=args.max_seq_length,
            split=args.split,
            split_index=subsets[subset.lower()],
            num_accumulation_steps=args.num_accumulation_steps,
            seed=args.seed,
        )
        return data_loader

    def __init__(self, dataset, num_samples, batch_size, max_seq_length=1024, split=[1,0,0], split_index=0, num_accumulation_steps=1, seed=1234):
        super().__init__()

        batch_size = batch_size // num_accumulation_steps
        self.reader = flow.nn.GPTIndexedBinDataReader(
            data_file_prefix=dataset,
            seq_length=max_seq_length,
            num_samples=num_samples,
            batch_size=batch_size,
            dtype=flow.int64,
            shuffle=True,
            random_seed=seed,
            split_sizes=split,
            split_index=split_index,
            placement=dist.get_layer_placement(0, "cpu"),
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
        )
        self.data_decoder = DataDecoder()
        self.label_decoder = LabelDecoder()

    def forward(self):
        tokens = self.reader()
        data = self.data_decoder(tokens)
        labels = self.label_decoder(tokens)
        return data, labels


class DataDecoder(flow.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tokens):
        assert tokens.ndim == 2
        return tokens.to_consistent(placement=dist.get_layer_placement(0))[:, :-1]


class LabelDecoder(flow.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tokens):
        assert tokens.ndim == 2
        return tokens.to_consistent(placement=dist.get_layer_placement(-1))[:, 1:]
