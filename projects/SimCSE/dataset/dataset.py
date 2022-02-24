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
import csv
from oneflow.utils.data import Dataset, DataLoader
from libai.tokenizer import BertTokenizer
import oneflow as flow
from libai.data.structures import DistTensorData, Instance


def load_data(name, path):
    assert name in ['nli', 'wiki', 'sts']
    
    def load_nli_data(path):
        # sup task
        csv_file = csv.reader(open(path, 'r', encoding='utf8'))
        data = [i for i in csv_file][1:]
        random.shuffle(data)
        # sent1, sent2, hard_neg
        return data
    
    def load_wiki_data(path):
        # unsup task
        data = []
        with open(path, 'r', encoding='utf8') as file:
            for line in file.readlines():
                line = ' '.join(line.strip().split())
                data.append(line)
        random.shuffle(data)
        return data

    def load_sts_data(path):
        # test
        data = []
        with open(path, 'r', encoding='utf8') as file:
            for line in file.readlines():
                line = line.strip().split('\t')
                data.append(line)
        return data
    
    if name == 'nli':
        return load_nli_data(path)
    elif name == 'wiki':
        return load_wiki_data(path)
    else:
        return load_sts_data(path)


class TrainDataset(Dataset):
    # unsup
    def __init__(self, name, path, tokenizer, max_len):
        self.name = name
        self.data = load_data(name, path)
        self.tokenizer = tokenizer
        self.max_len = int(max_len / 2)
        self.pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)

    def __len__(self):
        return len(self.data)
    
    def text2id(self, text):
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        ids = ids[:self.max_len-2]
        ids = [self.cls_id] + ids + [self.sep_id]
        
        attention_mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)
        
        return padding_for_ids(
            data = {"input_ids":ids, "attention_mask":attention_mask, "token_type_ids":token_type_ids}, 
            file_name = self.name,
            pad_id = self.pad_id,
            max_len = self.max_len
            )
    
    def __getitem__(self, index):
        # {"input_ids":ids, "attention_mask":attention_mask, "token_type_ids":token_type_ids}
        if self.name == 'wiki':
            return self.text2id(self.data[index])
        else:   # nli
            return self.text2id(self.data[index][0]), self.text2id(self.data[index][1]), self.text2id(self.data[index][2])


class TestDataset(Dataset):
    # sts datasets
    def __init__(self, name, path, tokenizer):
        self.data = load_data(name, path)
        self.tokenizer = tokenizer
        self.max_len = 50
        self.pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
    
    def __len__(self):
        return len(self.data)
    
    def text2id(self, text):
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = ids[:self.max_len-2]

        ids = [self.cls_id] + ids + [self.sep_id]
        length = len(ids)
        
        ids = ids + [self.pad_id] * (self.max_len - length)
        attention_mask = [1] * length + [self.pad_id] * (self.max_len - length)
        token_type_ids = [0] * self.max_len

        return {
            "input_ids" : ids,
            "attention_mask" : attention_mask,
            "token_type_ids" : token_type_ids
        }

    def __getitem__(self, index):
        # sent1, sent2, laebl
        sent1 = self.text2id(self.data[index][1])
        sent2 = self.text2id(self.data[index][2])
        score = float(self.data[index][0])
        return Instance(
            input_ids = DistTensorData(flow.tensor([sent1['input_ids'], sent2['input_ids']], dtype=flow.long)),
            attention_mask = DistTensorData(flow.tensor([sent1['attention_mask'], sent2['attention_mask']], dtype=flow.long)),
            token_type_ids = DistTensorData(flow.tensor([sent1['token_type_ids'], sent2['token_type_ids']], dtype=flow.long)),
            labels = DistTensorData(flow.tensor(score, dtype=flow.float))
        )
        

def padding_for_ids(data, file_name, pad_id=0, max_len=256):
    data['input_ids'] = data['input_ids'] + [pad_id] * (max_len - len(data['input_ids']))
    data['attention_mask'] = data['attention_mask'] + [pad_id] * (max_len - len(data['attention_mask']))
    data['token_type_ids'] = data['token_type_ids'] + [pad_id] * (max_len - len(data['token_type_ids']))

    if file_name == 'wiki':
        data['input_ids'] = [data['input_ids'], data['input_ids']]
        data['attention_mask'] = [data['attention_mask'], data['attention_mask']]
        data['token_type_ids'] = [data['token_type_ids'], data['token_type_ids']]
    
    return Instance(
        input_ids = DistTensorData(flow.tensor(data['input_ids'], dtype=flow.long)),
        attention_mask = DistTensorData(flow.tensor(data['attention_mask'], dtype=flow.long)),
        token_type_ids = DistTensorData(flow.tensor(data['token_type_ids'], dtype=flow.long))
    )
