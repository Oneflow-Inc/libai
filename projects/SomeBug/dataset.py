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

import csv
import random

import jsonlines
import oneflow as flow
from oneflow.utils.data import DataLoader, Dataset

from libai.data.structures import DistTensorData, Instance
from libai.tokenizer import BertTokenizer


def load_data(name, path):
    assert name in ["snli-sup", "snli-unsup", "lqcmc", "eng_sts", "cnsd_sts", "wiki", "add"]

    def load_snli_data_sup(path):
        with jsonlines.open(path, 'r') as f:
            return [(line['origin'], line['entailment'], line['contradiction']) for line in f]

    def load_cnsd_sts_data(path):
        with open(path, "r", encoding="utf8") as f:
            return [(line.split("||")[1], line.split("||")[2], line.split("||")[3]) for line in f]

    if name == "snli-sup":
        return load_snli_data_sup(path)
    elif name == "cnsd_sts":
        return load_cnsd_sts_data(path)

class TrainDataset_sup(Dataset):
    def __init__(self, name, path, tokenizer, max_len=64):
        self.data = load_data(name, path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = self.tokenizer.pad_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
    
    def __len__(self):
        return len(self.data)

    def pad_text(self, ids):
        attention_mask = [1] * len(ids)
        ids = ids + [self.pad_id] * (self.max_len - len(ids))
        attention_mask = attention_mask + [self.pad_id] * (self.max_len - len(attention_mask))
        return ids, attention_mask
    
    def text2id(self, text):
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = ids[: self.max_len - 2]
        ids = [self.cls_id] + ids + [self.sep_id]
        ids, attention_mask = self.pad_text(ids)
        return ids, attention_mask
    
    def __getitem__(self, index):
        ids0, mask0 = self.text2id(self.data[index][0])
        ids1, mask1 = self.text2id(self.data[index][1])
        ids2, mask2 = self.text2id(self.data[index][2])
        return Instance(
            input_ids = DistTensorData(
                flow.tensor([ids0, ids1, ids2], dtype=flow.long)
            ),
            attention_mask = DistTensorData(
                flow.tensor([mask0, mask1, mask2], dtype=flow.long)
            )
        )

class TestDataset_sup(TrainDataset_sup):
    def __getitem__(self, index):
        label = int(self.data[index][2])
        ids0, mask0 = self.text2id(self.data[index][0])
        ids1, mask1 = self.text2id(self.data[index][1])
        return Instance(
            input_ids = DistTensorData(
                flow.tensor([ids0, ids1], dtype=flow.long)
            ),
            attention_mask = DistTensorData(
                flow.tensor([mask0, mask1], dtype=flow.long)
            ),
            labels = DistTensorData(
                flow.tensor(label, dtype=flow.int)
            ),
        )