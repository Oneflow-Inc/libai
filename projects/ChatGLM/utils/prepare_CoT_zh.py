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
import json
import os
import random

random.seed(2023)

DATA_DIR = os.environ["DATA_DIR"]
csv_file = os.path.join(DATA_DIR, "CoT_zh/CoT_Chinese_data.csv")
train_json_file = os.path.join(DATA_DIR, "CoT_zh/train.json")
test_json_file = os.path.join(DATA_DIR, "CoT_zh/test.json")

train = {"prompt": [], "query": [], "response": []}
test = {"prompt": [], "query": [], "response": []}
with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        if random.random() < 0.99:  # for train
            train["prompt"].append(line[0])
            train["query"].append(line[1])
            train["response"].append(line[2])
        else:
            test["prompt"].append(line[0])
            test["query"].append(line[1])
            test["response"].append(line[2])

with open(train_json_file, "w", encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=4)
with open(test_json_file, "w", encoding="utf-8") as f:
    json.dump(test, f, ensure_ascii=False, indent=4)
