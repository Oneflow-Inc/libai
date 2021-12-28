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

from .bert_dataset import BertDataset
from .gpt_dataset import GPT2Dataset
from .data_utils import get_indexed_dataset, get_prefixes_and_weights
from .split_dataset import split_ds
from .blendable_dataset import BlendableDataset

def build_single_dataset(args, tokenizer, data_prefix, data_impl, split=None, skip_warmup=False, data_type='gpt'):
    indexed_dataset = get_indexed_dataset(data_prefix, data_impl, skip_warmup=skip_warmup)
    if data_type == 'bert':
        total_dataset = BertDataset(tokenizer, data_prefix, indexed_dataset, max_seq_length=args.max_seq_length, 
                                    mask_lm_prob=args.mask_lm_prob, binary_head=args.binary_head)
    else:
        total_dataset = GPT2Dataset(tokenizer, data_prefix, indexed_dataset, max_seq_length=args.max_seq_length)
    train_dataset, valid_dataset, test_dataset = split_ds(total_dataset, split)
    return train_dataset, valid_dataset, test_dataset

def build_dataset(args, tokenizer, data_prefix, data_impl, split=None, skip_warmup=False, data_type='gpt'):
    if len(data_prefix) == 1:
        return build_single_dataset(args, tokenizer, data_prefix[0], data_impl, split, skip_warmup, data_type)
    
    prefixes, weights = get_prefixes_and_weights(data_prefix)
    train_datasets, valid_datasets, test_datasets = [], [], []
    for prefix in range(prefixes):
        train_dataset, valid_dataset, test_dataset = build_single_dataset(args, tokenizer, prefix, data_impl, split, skip_warmup, data_type)
        if train_dataset:
            train_datasets.append(train_dataset)
        if valid_dataset:
            valid_datasets.append(valid_dataset)
        if test_dataset:
            test_datasets.append(test_dataset)
    
    blending_train_dataset, blending_valid_dataset, blending_test_dataset = None, None, None
    if len(train_datasets) > 0:
        blending_train_dataset = BlendableDataset(train_datasets, weights)
    if len(valid_datasets) > 0:
        blending_valid_dataset = BlendableDataset(valid_datasets, weights)
    if len(test_datasets) > 0:
        blending_test_dataset = BlendableDataset(test_datasets, weights)
    
    return blending_train_dataset, blending_valid_dataset, blending_test_dataset 

