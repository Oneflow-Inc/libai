

import copy
import json

import oneflow as flow
from oneflow.utils.data import Dataset

from libai.data.structures import DistTensorData, Instance


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class AlpacaDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=513, partition="train"):
        self.data = json.load(open(path))
        self.tokenizer = tokenizer
        self.max_len = max_len
        if partition == "train":
            self.data = self.data
        else:
            self.data = self.data[:200]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if data.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(data)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(data)
        example = prompt + data["output"]
        prompt = self.tokenizer.tokenize(prompt, add_bos=True, add_eos=False, device="cpu")[0]
        example = self.tokenizer.tokenize(example, add_bos=True, add_eos=True, device="cpu")[0]
        padding = self.max_len - example.shape[0]
        if padding > 0:
            example = flow.cat((example, flow.zeros(padding, dtype=flow.long) - 1))
        elif padding < 0:
            example = example[: self.max_len]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = -1
        example = example[:-1]
        labels = labels[1:]
        example_mask = flow.where(example_mask, flow.tensor(0, dtype=flow.float), flow.tensor(-float('inf')))
        example_mask = example_mask[:-1]
        return Instance(
            input_ids=DistTensorData(example),
            labels=DistTensorData(labels),
        )
