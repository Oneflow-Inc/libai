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

import random

import jsonlines
import oneflow as flow
from oneflow.utils.data import Dataset

from libai.data.structures import DistTensorData, Instance


def load_data(name, path):
    assert name in ["snli-sup", "snli-unsup", "lqcmc", "eng_sts", "cnsd_sts", "wiki", "add"]

    def load_snli_data_unsup(path):
        with jsonlines.open(path, "r") as f:
            return [line.get("origin") for line in f]

    def load_snli_data_sup(path):
        with jsonlines.open(path, "r") as f:
            return [(line["origin"], line["entailment"], line["contradiction"]) for line in f]

    def load_lqcmc_data(path):
        with open(path, "r", encoding="utf8") as f:
            return [line.strip().split("\t")[0] for line in f]

    def load_cnsd_sts_data(path):
        with open(path, "r", encoding="utf8") as f:
            return [(line.split("||")[1], line.split("||")[2], line.split("||")[3]) for line in f]

    def load_wiki_data(path):
        data = []
        with open(path, "r", encoding="utf8") as file:
            for line in file.readlines():
                line = " ".join(line.strip().split())
                data.append(line)
        return data

    def load_eng_sts_data(path):
        data = []
        with open(path, "r", encoding="utf8") as file:
            for line in file.readlines():
                line = line.strip().split("\t")
                data.append(line)
        return data

    def load_sts_to_train(path):
        if path is None:
            return []
        with open(
            path,
            "r",
            encoding="utf8",
        ) as f:
            data = [line.split("||")[1] for line in f]
        return data

    if name == "snli-unsup":
        return load_snli_data_unsup(path)
    elif name == "snli-sup":
        return load_snli_data_sup(path)
    elif name == "wiki":
        return load_wiki_data(path)
    elif name == "cnsd_sts":
        return load_cnsd_sts_data(path)
    elif name == "eng_sts":
        return load_eng_sts_data(path)
    elif name == "lqcmc":
        return load_lqcmc_data(path)
    else:
        return load_sts_to_train(path)


def padding_for_ids(data, pad_id=0, max_len=64):
    data["input_ids"] = data["input_ids"] + [pad_id] * (max_len - len(data["input_ids"]))
    data["attention_mask"] = data["attention_mask"] + [0] * (max_len - len(data["attention_mask"]))

    data["input_ids"] = [data["input_ids"], data["input_ids"]]
    data["attention_mask"] = [data["attention_mask"], data["attention_mask"]]

    return Instance(
        input_ids=DistTensorData(flow.tensor(data["input_ids"], dtype=flow.long)),
        attention_mask=DistTensorData(flow.tensor(data["attention_mask"], dtype=flow.long)),
    )


class TrainDataset_unsup(Dataset):
    # unsup
    def __init__(self, name, path, tokenizer, max_len, path2=None):
        self.name = name
        self.data = load_data(name, path) + load_data("add", path2)
        random.shuffle(self.data)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = self.tokenizer.pad_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id

    def __len__(self):
        return len(self.data)

    def text2id(self, text):
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        ids = ids[: self.max_len - 2]
        ids = [self.cls_id] + ids + [self.sep_id]

        attention_mask = [1] * len(ids)

        return padding_for_ids(
            data={
                "input_ids": ids,
                "attention_mask": attention_mask,
            },
            pad_id=self.pad_id,
            max_len=self.max_len,
        )

    def __getitem__(self, index):
        return self.text2id(self.data[index])


class TestDataset_unsup(Dataset):
    # sts datasets
    def __init__(self, name, path, tokenizer):
        self.data = load_data(name, path)
        self.tokenizer = tokenizer
        self.max_len = 64
        self.pad_id = self.tokenizer.pad_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id

    def __len__(self):
        return len(self.data)

    def text2id(self, text):
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = ids[: self.max_len - 2]

        ids = [self.cls_id] + ids + [self.sep_id]
        length = len(ids)

        ids = ids + [self.pad_id] * (self.max_len - length)
        attention_mask = [1] * length + [0] * (self.max_len - length)

        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, index):
        # sent1, sent2, laebl
        sample = self.data[index]

        sent1 = self.text2id(sample[0])
        sent2 = self.text2id(sample[1])
        score = int(sample[2])

        return Instance(
            input_ids=DistTensorData(
                flow.tensor([sent1["input_ids"], sent2["input_ids"]], dtype=flow.long)
            ),
            attention_mask=DistTensorData(
                flow.tensor([sent1["attention_mask"], sent2["attention_mask"]], dtype=flow.long)
            ),
            labels=DistTensorData(flow.tensor(score, dtype=flow.int)),
        )


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
        attention_mask = attention_mask + [0] * (self.max_len - len(attention_mask))
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
            input_ids=DistTensorData(flow.tensor([ids0, ids1, ids2], dtype=flow.long)),
            attention_mask=DistTensorData(flow.tensor([mask0, mask1, mask2], dtype=flow.long)),
        )


class TestDataset_sup(TrainDataset_sup):
    def __getitem__(self, index):
        label = int(self.data[index][2])
        ids0, mask0 = self.text2id(self.data[index][0])
        ids1, mask1 = self.text2id(self.data[index][1])
        return Instance(
            input_ids=DistTensorData(flow.tensor([ids0, ids1], dtype=flow.long)),
            attention_mask=DistTensorData(flow.tensor([mask0, mask1], dtype=flow.long)),
            labels=DistTensorData(flow.tensor(label, dtype=flow.int)),
        )
