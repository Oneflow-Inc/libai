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

import json
from typing import Dict, List

import oneflow as flow
from oneflow.utils.data import Dataset

from libai.data.structures import DistTensorData, Instance
from libai.utils import distributed as dist
from libai.utils.logger import setup_logger

IGNORE_INDEX = -100
logger = setup_logger()


class ChatGLMTrainDataset(Dataset):
    def __init__(self, path, tokenizer, max_source_len=128, max_target_len=128, max_length=None):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        if max_length is None:
            self.max_len = max_source_len + max_target_len + 1
        else:
            self.max_len = max_length

        example = self._preprocess(0)
        self.log_dataset_example(example)

    def _preprocess(self, idx):
        # inputs with format `<bos> X Y <eos>` labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.

        item = {key: self.data[key][idx] for key in self.data}
        # prompt, query, response

        source_ids = self.tokenizer.encode(item["prompt"] + item["query"], add_special_tokens=True)[
            0
        ]
        source_ids = source_ids[: self.max_source_len]

        target_ids = self.tokenizer.encode(item["response"], add_special_tokens=False)[0]
        target_ids = target_ids[: self.max_target_len]

        input_ids = source_ids + target_ids
        labels = [self.tokenizer.pad_token_id] * len(source_ids) + target_ids

        input_ids = input_ids[: self.max_len - 1] + [self.tokenizer.eos_token_id]
        labels = labels[: self.max_len - 1] + [self.tokenizer.eos_token_id]

        # left pad
        pad_len = self.max_len - len(input_ids)
        input_ids = [self.tokenizer.pad_token_id] * pad_len + input_ids
        labels = [self.tokenizer.pad_token_id] * pad_len + labels
        labels = [(l if l != self.tokenizer.pad_token_id else IGNORE_INDEX) for l in labels]

        return {"input_ids": input_ids, "labels": labels}

    def log_dataset_example(self, example: Dict[str, List[int]]) -> None:
        if dist.is_main_process():
            logger.info("input_ids:\n{}".format(example["input_ids"]))
            logger.info(
                "inputs:\n{}".format(
                    self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)
                )
            )
            logger.info("label_ids:\n{}".format(example["labels"]))
            logger.info(
                "labels:\n{}".format(
                    self.tokenizer.decode(
                        list(filter(lambda x: x != IGNORE_INDEX, example["labels"])),
                        skip_special_tokens=False,
                    )
                )
            )

    def __len__(self):
        return len(self.data["prompt"])

    def __getitem__(self, index):
        item = self._preprocess(index)
        return Instance(
            input_ids=DistTensorData(flow.LongTensor(item["input_ids"])),
            labels=DistTensorData(flow.LongTensor(item["labels"]), placement_idx=-1),
        )
