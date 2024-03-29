import os
import json
from tqdm import tqdm
import random

import oneflow as flow


IGNORE_TOKEN_ID = -100

data = {
    'id': 'i6IyJda_0', 
    'conversations': [
        {'from': 'human', 'value': 'How to tell if a customer segment is well segmented? In 3 bullet points.'}, 
        {'from': 'gpt', 'value': '1. Homogeneity \n2. Distinctiveness \n3. Stability'},
        {'from': 'human', 'value': 'Thank you'}, 
        {'from': 'gpt', 'value': 'you are welcome'}, 
    ]
}


def qwen2_data_process(
    sources,
    tokenizer,
    system_message: str = "You are a helpful assistant.",
):
    max_len = tokenizer.model_max_length
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.encode("<|im_start|>")[0]
    im_end = tokenizer.encode("<|im_end|>")[0]
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = (
            [im_start]
            + _system
            + tokenizer(system_message).input_ids
            + [im_end]
            + nl_tokens
        )
        input_id += system
        target += (
            [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        )
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = (
                tokenizer(role).input_ids
                + nl_tokens
                + tokenizer(sentence["value"]).input_ids
                + [im_end]
                + nl_tokens
            )
            input_id += _input_id
            if role == "<|im_start|>user":
                _target = (
                    [im_start]
                    + [IGNORE_TOKEN_ID] * (len(_input_id) - 3)
                    + [im_end]
                    + nl_tokens
                )
            elif role == "<|im_start|>assistant":
                _target = (
                    [im_start]
                    + [IGNORE_TOKEN_ID] * (len(tokenizer(role).input_ids) - 1)
                    + _input_id[len(tokenizer(role).input_ids) : -2]
                    + [im_end]
                    + nl_tokens
                )
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = flow.tensor(input_ids, dtype=flow.int, device="cpu")
    targets = flow.tensor(targets, dtype=flow.long, device="cpu")
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    attention_mask = flow.where(attention_mask, flow.tensor(0.0), flow.tensor(-float("Inf")))

    return dict(
        input_ids=input_ids[0],
        labels=targets[0],
        attention_mask=attention_mask[0],
    )


def preprocess(input_file, targe_file, shuffle=False, tokenizer=None):
    file = open(input_file, "r")
    data = json.load(file)
    if shuffle:
        random.shuffle(data)
    train_set = [qwen2_data_process([sample["conversations"]], tokenizer) for sample in tqdm(data)]
    flow.save(train_set, os.path.join(targe_file, "train_set"))
    print("training dataset saved in {}\n".format(os.path.join(targe_file, "train_set")))


if __name__ == "__main__":
  
    from projects.mock_transformers.mock_tokenization import Qwen2Tokenizer

    input_file = "/data/home/xiezipeng/libai/projects/Qwen/subset.json"
    target_file = "/data/home/xiezipeng/libai/projects/Qwen"
    model_file = "/data/home/xiezipeng/hf_models/Qwen/Qwen1.5-7B"
    
    tokenizer = Qwen2Tokenizer.from_pretrained(model_file)
    tokenizer.model_max_length = 2048

    preprocess(
        input_file=input_file,
        targe_file=target_file, 
        tokenizer=tokenizer
    )

    # res = qwen2_data_process([data["conversations"]], tokenizer)
    # input_ids = res["input_ids"]
    # labels = res["labels"]
    # attention_mask = res["attention_mask"]

    # print(input_ids[0])
    # print(labels)
    # print(attention_mask)

    # labels = labels[0]
    # labels[labels==IGNORE_TOKEN_ID] = 151643
    
    # print("input text:\n",tokenizer.decode(input_ids[0].tolist()))
    # print("labels text: \n",tokenizer.decode(labels.tolist()))
