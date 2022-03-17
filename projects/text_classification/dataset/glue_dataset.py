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
import os
import time
from enum import Enum
from typing import Optional, Union

import oneflow as flow
from filelock import FileLock
from oneflow.utils.data import Dataset

from libai.data.structures import DistTensorData, Instance

from .utils import EncodePattern
from .utils_glue import glue_convert_examples_to_features, glue_output_modes, glue_processors

logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class GlueDataset(Dataset):
    def __init__(
        self,
        task_name,
        data_dir,
        tokenizer,
        max_seq_length: int = 128,
        mode: Union[str, Split] = Split.train,
        pattern: Union[str, EncodePattern] = EncodePattern.bert_pattern,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False,
    ):
        self.processor = glue_processors[task_name]()
        self.output_mode = glue_output_modes[task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        if isinstance(pattern, str):
            try:
                pattern = EncodePattern[pattern]
            except KeyError:
                raise KeyError("pattern is not a valid pattern method")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else data_dir,
            f"cached_{mode.value}_{tokenizer.__class__.__name__}_{max_seq_length}_{task_name}",
        )
        label_list = self.processor.get_labels()
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                self.features = flow.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]",
                    time.time() - start,
                )
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(data_dir)
                else:
                    examples = self.processor.get_train_examples(data_dir)

                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=max_seq_length,
                    pattern=pattern,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                flow.save(self.features, cached_features_file)
                logger.info(
                    f"Saving features into cached file {cached_features_file} "
                    f"[took {time.time() - start:.3f} s]"
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        feature = self.features[i]
        tensors = {}
        for k, v in feature.__dict__.items():
            if v is not None:
                if k == "label":
                    dtype = flow.long if isinstance(v, int) else flow.float
                    t = flow.tensor(v, dtype=dtype)
                    tensors[k] = DistTensorData(t, placement_idx=-1)
                else:
                    t = flow.tensor(v, dtype=flow.long)
                    tensors[k] = DistTensorData(t)
        sample = Instance(**tensors)
        return sample

    def get_labels(self):
        return self.label_list
