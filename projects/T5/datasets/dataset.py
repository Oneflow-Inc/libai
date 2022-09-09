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

import warnings

try:
    import datasets
except:  # noqa
    warnings.warn("datasets library is needed")
import numpy as np
import oneflow as flow
from oneflow.utils.data import Dataset

from libai.data.structures import DistTensorData, Instance


def get_data(path):
    total_data = []
    for i in range(10):
        path = path[:-1] + str(i)
        dataset = datasets.load_from_disk(path)
        for i in dataset:
            total_data.append(i)
    return total_data


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/
    text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/
    t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs
    sentinel tokens, and each non-noise span in the targets is replaced by
    extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


class UnsuperviseT5Dataset(Dataset):
    """This function is copy of https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/
    ec13aeb8689cfafaa6a7a9e9595d110edbe34123/fengshen/data/t5_dataloader/t5_datasets.py#L61.
    """

    def __init__(self, data_path):
        # [{input_ids: ...}, {input_ids: ...}, ...]
        self.data = get_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return x


class collate_fn:
    def __init__(
        self,
        vocab_size,
        max_seq_length,
        noise_density,
        mean_noise_span_length,
        eos_token_id=1,
        pad_token_id=0,
        decoder_start_token_id=0,
    ):
        self.max_seq_length = max_seq_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.vocab_size = vocab_size
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.expanded_inputs_length, self.targets_length = compute_input_and_target_lengths(
            inputs_length=self.max_seq_length,
            noise_density=self.noise_density,
            mean_noise_span_length=self.mean_noise_span_length,
        )

    def __call__(self, examples):
        batch = {
            k: np.array([examples[i][k] for i in range(len(examples))])
            for k, v in examples[0].items()
        }
        input_ids = np.array(batch["input_ids"])
        batch_size, expanded_input_length = input_ids.shape
        mask_indices = np.asarray(
            [self.random_spans_noise_mask(expanded_input_length) for i in range(batch_size)]
        )
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.max_seq_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is \
                    {batch['input_ids'].shape[-1]}, but should be {self.targets_length}."
            )

        if batch["labels"].shape[-1] != self.targets_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is \
                    {batch['labels'].shape[-1]}, but should be {self.targets_length}."
            )

        batch["decoder_input_ids"] = self.shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        return Instance(
            encoder_input_ids=DistTensorData(flow.tensor(batch["input_ids"])),
            decoder_input_ids=DistTensorData(flow.tensor(batch["decoder_input_ids"])),
            encoder_attn_mask=DistTensorData(
                flow.ones(len(batch["input_ids"]), len(batch["input_ids"][0])).to(flow.bool)
            ),
            decoder_attn_mask=DistTensorData(
                flow.ones(len(batch["decoder_input_ids"]), len(batch["decoder_input_ids"][0])).to(
                    flow.bool
                )
            ),
            encoder_decoder_attn_mask=DistTensorData(
                flow.ones(
                    len(batch["input_ids"]),
                    len(batch["decoder_input_ids"][0]),
                    len(batch["input_ids"][0]),
                ).to(flow.bool)
            ),
            lm_labels=DistTensorData(flow.tensor(batch["labels"])),
            loss_mask=DistTensorData(flow.tensor(batch["labels"])),
        )

    def filter_input_ids(self, input_ids, sentinel_ids):
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def create_sentinel_ids(self, mask_indices):
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(sentinel_ids != 0, (self.vocab_size - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def random_spans_noise_mask(self, length):
        orig_length = length
        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

    def shift_tokens_right(
        self, input_ids: np.array, pad_token_id: int, decoder_start_token_id: int
    ) -> np.ndarray:
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = np.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        shifted_input_ids[:, 0] = decoder_start_token_id

        shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
        return shifted_input_ids
