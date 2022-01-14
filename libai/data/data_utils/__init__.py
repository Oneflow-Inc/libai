# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

from .indexed_dataset import get_indexed_dataset, IndexedDataset, IndexedCachedDataset, MMapIndexedDataset
from .blendable_dataset import BlendableDataset
from .data_utils import get_prefixes_and_weights
from .reindexed_dataset import SentenceIndexedDataset, BlockIndexedDataset
from .split_dataset import split_ds, SplitDataset
