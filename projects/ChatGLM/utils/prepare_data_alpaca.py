"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import json
import math
import os
from pathlib import Path

import oneflow as flow
import requests
from oneflow.utils.data import random_split
from tqdm import tqdm

from libai.utils.logger import setup_logger

logger = setup_logger()


def prepare(
    destination_path: Path = Path(os.environ["DATA_DIR"]),
    test_split_fraction: float = 0.03865,  # to get exactly 2000 test samples,
    seed: int = 42,
    data_file_name: str = "alpaca_data_cleaned_archive.json",
    data_file_url: str = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json",  # noqa
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    destination_path.mkdir(parents=True, exist_ok=True)
    data_file_path = destination_path / data_file_name
    logger.info("Loading data file...")
    download_if_missing(data_file_path, data_file_url)
    with open(data_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

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
    train_set = [prepare_sample(example=sample) for sample in tqdm(train_set)]

    logger.info("Processing test split ...")
    test_set = [prepare_sample(example=sample) for sample in tqdm(test_set)]
    train_file_path = destination_path / "train.json"
    test_file_path = destination_path / "test.json"

    train_data = trans_data_format(train_set)
    test_data = trans_data_format(test_set)
    with open(train_file_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4)
    with open(test_file_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)


def trans_data_format(data):
    assert len(data) > 0
    keys = data[0].keys()
    new_data = {key: [] for key in keys}
    for item in data:
        for key in keys:
            new_data[key].append(item[key])
    return new_data


def download_if_missing(file_path: Path, file_url: str) -> None:
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists() and file_path.stat().st_size > 0:
        return
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(file_url).text)


def prepare_sample(example: dict) -> dict:
    """Processes a single sample."""
    full_prompt = generate_prompt(example)

    ans = {"prompt": full_prompt, "query": "", "response": example["output"]}

    return ans


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
