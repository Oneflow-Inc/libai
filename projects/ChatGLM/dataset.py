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
import sys
sys.path.append(
    '/home/lixin/codes/libai/projects/ChatGLM'
)

from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Tuple, Union
import random
from transformers import Trainer
import oneflow as flow
import json
from oneflow.utils.data import Dataset

from libai.data.structures import DistTensorData, Instance
from libai.utils.logger import setup_logger
from template import get_template_and_fix_tokenizer

IGNORE_INDEX = -100

logger = setup_logger()

class ChatGLMTrainDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=1350):
        with open(path, 'r',encoding='utf-8') as f:
            self.data = json.load(f)
        self.template = get_template_and_fix_tokenizer('chatglm3', tokenizer)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_inputs = self._preprocess()
        example = {key: self.model_inputs[key][0] for key in self.model_inputs}
        self.log_supervised_dataset_example(example)

    @staticmethod
    def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
        for i in range(len(examples["prompt"])):
            query, response = examples["prompt"][i], examples["response"][i]
            query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
            history = examples["history"][i] if "history" in examples else None
            system = examples["system"][i] if "system" in examples else None
            yield query, response, history, system

    @staticmethod
    def infer_max_len(source_len: int, target_len: int, max_length:int) -> Tuple[int, int]:
        max_target_len = int(max_length * (target_len / (source_len + target_len)))
        max_source_len = max_length - max_target_len
        return max_source_len, max_target_len

    def _preprocess(self):
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in ChatGLMTrainDataset.construct_example(self.data):
            if not (isinstance(query, str) and isinstance(response, str) and query != "" and response != ""):
                continue

            input_ids, labels = [], []
            for turn_idx, (source_ids, target_ids) in enumerate(self.template.encode_multiturn(
                self.tokenizer, query, response, history, system
            )):
                source_len, target_len = len(source_ids), len(target_ids)
                max_source_len, max_target_len = ChatGLMTrainDataset.infer_max_len(source_len, target_len, self.max_len)
                if source_len > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if target_len > max_target_len:
                    target_ids = target_ids[:max_target_len]

                if turn_idx != 0 and self.template.efficient_eos:
                    source_mask = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

            if self.template.efficient_eos:
                input_ids += [self.tokenizer.eos_token_id]
                labels += [self.tokenizer.eos_token_id]

            if len(input_ids) > self.max_len:
                input_ids = input_ids[:self.max_len]
                labels = labels[:self.max_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def log_supervised_dataset_example(self, example: Dict[str, List[int]]) -> None:
        logger.info("input_ids:\n{}".format(example["input_ids"]))
        logger.info("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        logger.info("label_ids:\n{}".format(example["labels"]))
        logger.info("labels:\n{}".format(
            self.tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        ))

    def __len__(self):
        return len(self.data['prompt'])

    def __getitem__(self, index):
        
        input_ids = self.model_inputs['input_ids'][index]
        attention_mask = self.model_inputs['attention_mask'][index]
        labels = self.model_inputs['labels'][index]

        return Instance(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    @staticmethod
    def collate_fn(instance_lists: List["Instance"]) -> "Instance":
        assert all(isinstance(i, Instance) for i in instance_lists)
        assert len(instance_lists) > 0

        key2pad = {
            'input_ids': 0,
            'attention_mask':0,
            'labels':IGNORE_INDEX
        }
        # input_ids,attention_mask,labels
        ret = Instance()
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            # max_len = max([len(v) for v in values])
            max_len = 256
            for i,value in enumerate(values):
                if len(value) < max_len:
                    value.extend([key2pad[k] for _ in range(max_len - len(value))])
                else:
                    values[i] = values[i][:max_len]

            if k == "labels":
                values = flow.tensor(values, dtype=flow.long)
                values = DistTensorData(values)
            elif k == "attention_mask":
                values = flow.tensor(values, dtype=flow.bool)
                values = DistTensorData(values)
            else:
                values = flow.tensor(values, dtype=flow.long)
                values = DistTensorData(values)
            # print('aaaa',values.tensor.size())
            ret.set(k, values)
        return ret


if __name__ == '__main__':
    from tokenizer import ChatGLMTokenizer
    from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
    tokenizer = ChatGLMTokenizer(
        '/home/lixin/.cache/modelscope/hub/ZhipuAI/chatglm3-6b/tokenizer.model'
    )
    dataset = ChatGLMTrainDataset(
        '/home/lixin/DATA/CoT_zh/train.json',
        tokenizer
    )

    dataloader,_,_ = build_nlp_train_loader(
        dataset = [dataset],
        train_batch_size = 4,
        collate_fn=ChatGLMTrainDataset.collate_fn
    )

    for data in dataloader:
        print(data)
        ret_dict = {}
        for key, value in data.get_fields().items():
            value.to_global(device_type='cuda')
            ret_dict[key] = value.tensor
        print()
