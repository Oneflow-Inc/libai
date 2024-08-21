"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import copy
import json
import math
import os
from pathlib import Path
from typing import Optional

import oneflow as flow
import requests
from oneflow.utils.data import random_split
from tqdm import tqdm

from libai.config import instantiate
from libai.utils.logger import setup_logger
from projects.Llama.configs.llama_config import tokenization

logger = setup_logger()


def prepare(
    destination_path: Path = Path("alpaca_data"),
    checkpoint_dir: Path = Path("meta-llama/Llama-2-7b-hf"),
    test_split_fraction: float = 0.03865,  # to get exactly 2000 test samples,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    data_file_name: str = "alpaca_data_cleaned_archive.json",
    data_file_url: str = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json",  # noqa
    ignore_index: int = -1,
    max_seq_length: Optional[int] = 512,
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.
    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    if max_seq_length is None:
        with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as file:
            config = json.load(file)
            max_seq_length = config["max_position_embeddings"]

    destination_path.mkdir(parents=True, exist_ok=True)
    data_file_path = destination_path / data_file_name
    logger.info("Loading data file...")
    download_if_missing(data_file_path, data_file_url)
    with open(data_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    logger.info("Loading tokenizer...")
    tokenizer = instantiate(tokenization.tokenizer)

    # Partition the dataset into train and test
    num_of_test_samples = math.floor(test_split_fraction * len(data))
    num_of_train_samples = len(data) - num_of_test_samples
    train_set, test_set = random_split(
        data,
        [num_of_train_samples, num_of_test_samples],
        generator=flow.Generator().manual_seed(seed),
    )
    train_set, test_set = list(train_set), list(test_set)

    logger.info(f"train has {len(train_set):,} samples")
    logger.info(f"test has {len(test_set):,} samples")

    logger.info("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
        )
        for sample in tqdm(train_set)
    ]
    flow.save(train_set, destination_path / "train")

    logger.info("Processing test split ...")
    test_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
        )
        for sample in tqdm(test_set)
    ]
    flow.save(test_set, destination_path / "test")

    max_length = max([i["input_ids"].shape[0] for i in train_set])
    logger.info("Max length of training dataset: {}".format(max_length))


def download_if_missing(file_path: Path, file_url: str) -> None:
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists() and file_path.stat().st_size > 0:
        return
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(file_url).text)


def prepare_sample(example: dict, tokenizer, max_length: int) -> dict:
    """Processes a single sample.
    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string
    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.
    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]

    prompt = tokenizer.tokenize(full_prompt, add_bos=True, add_eos=False, device="cpu")[0]
    example = tokenizer.tokenize(
        full_prompt_and_response, add_bos=True, add_eos=True, device="cpu"
    )[0]

    padding = max_length - example.shape[0]
    if padding > 0:
        example = flow.cat((example, flow.zeros(padding, dtype=flow.long) - 1))
    elif padding < 0:
        example = example[:max_length]
    labels = copy.deepcopy(example)
    labels[: len(prompt)] = -1
    example_mask = example.ge(0)
    label_mask = labels.ge(0)
    example[~example_mask] = 0
    labels[~label_mask] = -1
    example = example[:-1]
    labels = labels[1:]
    example_mask = flow.where(
        example_mask, flow.tensor(0, dtype=flow.float), flow.tensor(-float("inf"))
    )
    example_mask = example_mask[:-1]
    return {
        "input_ids": example,
        "labels": labels,
    }


def generate_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "  # noqa
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"  # noqa
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


if __name__ == "__main__":
    prepare()
