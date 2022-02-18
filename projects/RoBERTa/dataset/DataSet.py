import os
import json

import oneflow as flow
task_name = 'SST-2'  # CoLA or SST-2

def read_data(split):
    fn = os.path.join(task_name, split,
                      '{}.json'.format(split))
    input_ids = []
    attention_mask = []
    labels = []
    with open(fn, 'r') as f:
        result = json.load(f)
        for pack_data in result:
            input_ids.append(pack_data["input_ids"])
            # input_mask是什么？是说这个句子多长吗？
            attention_mask.append(pack_data["input_mask"])
            labels.append(pack_data["label_ids"])
    input_ids = flow.tensor(input_ids, dtype=flow.int32)
    attention_mask = flow.tensor(attention_mask, dtype=flow.int32)
    labels = flow.tensor(labels, dtype=flow.long)
    return input_ids, attention_mask, labels