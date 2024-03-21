import oneflow as flow


IGNORE_TOKEN_ID = -1

data = {
    'id': 'i6IyJda_0', 
    'conversations': [
        {'from': 'human', 'value': 'How to tell if a customer segment is well segmented? In 3 bullet points.'}, 
        {'from': 'gpt', 'value': '1. Homogeneity \n2. Distinctiveness \n3. Stability'},
        {'from': 'human', 'value': 'Thank you'}, 
        {'from': 'gpt', 'value': 'you are welcome'}, 
    ]
}


def preprocess_qwen2(
    sources,
    tokenizer,
    system_message: str = "You are a helpful assistant.",
):
    max_len = tokenizer.model_max_length
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start = tokenizer.encode("<|im_start|>")[0]
    im_end = tokenizer.encode("<|im_end|>")[0]
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
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
    input_ids = flow.tensor(input_ids, dtype=flow.int)
    targets = flow.tensor(targets, dtype=flow.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


if __name__ == "__main__":
    from projects.mock_transformers.mock_tokenization import Qwen2Tokenizer
    
    
    tokenizer = Qwen2Tokenizer.from_pretrained(
        "/data/home/xiezipeng/hf_models/Qwen/Qwen1.5-7B/",
    )
    tokenizer.model_max_length=110

    res = preprocess_qwen2([data["conversations"]], tokenizer)
    input_ids = res["input_ids"]
    labels = res["labels"]
    attention_mask = res["attention_mask"]

    # labels = labels[0]
    # labels[labels==-1] = 151643
    # print(input_ids[0])
    # print(labels)
    # print("input text:\n",tokenizer.decode(input_ids[0].tolist()))
    # print("labels text: \n",tokenizer.decode(labels.tolist()))
