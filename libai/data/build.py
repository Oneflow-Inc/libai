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

from .data_utils import get_indexed_dataset, get_prefixes_and_weights
from .data_utils import split_ds
from .data_utils import BlendableDataset

from libai.config import instantiate
from libai.utils.registry import Registry

DATASET_REGISTRY = Registry("dataset")
DATASET_REGISTRY.__doc__ = """
Registry for dataset, i.e. BertDataset.

The registered object will be called with `obj(cfg)` 
and expected to return a `Dataset` object.
"""

def build_dataset(cfg, tokenizer, data_prefix=None, split=None):
    """ Build dataset, defined by ``cfg``.
    """
    if data_prefix is None:
        data_prefix = cfg.data_prefix
    
    if isinstance(data_prefix, str):
        dataset_cls = DATASET_REGISTRY.get(cfg.dataset_name)
        indexed_dataset = get_indexed_dataset(data_prefix, cfg.data_impl, skip_warmup=cfg.skip_warmup)
        dataset = dataset_cls(tokenizer, data_prefix, indexed_dataset, **cfg.dataset_cls)
        train_dataset, valid_dataset, test_dataset = split_ds(total_dataset, split)
        return train_dataset, valid_dataset, test_dataset
    elif len(data_prefix) == 1:
        dataset_cls = DATASET_REGISTRY.get(cfg.dataset_name)
        indexed_dataset = get_indexed_dataset(data_prefix[0], cfg.data_impl, skip_warmup=cfg.skip_warmup)
        dataset = dataset_cls(tokenizer, data_prefix[0], indexed_dataset, **cfg.dataset_cls)
        train_dataset, valid_dataset, test_dataset = split_ds(total_dataset, split)
        return train_dataset, valid_dataset, test_dataset
    else:
        prefixes, weights = get_prefixes_and_weights(data_prefix)
        train_datasets, valid_datasets, test_datasets = [], [], []
        for prefix in range(prefixes):
            train_dataset, valid_dataset, test_dataset = build_dataset(cfg, tokenizer, data_prefix=prefix, split=split)
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

