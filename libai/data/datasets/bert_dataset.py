# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""BERT Style dataset."""

import numpy as np
import oneflow as flow

from libai.data.structures import DistTensorData, Instance

from ..data_utils import create_masked_lm_predictions, get_samples_mapping


class BertDataset(flow.utils.data.Dataset):
    """Dataset containing sentence pairs for BERT training.
    Each index corresponds to a randomly generated sentence pair.

    Args:
        name: Name of dataset for clarification.
        tokenizer: Tokenizer to use.
        data_prefix: Path to the training dataset.
        indexed_dataset: Indexed dataset to use.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length. Defaults to 512.
        mask_lm_prob: Probability to mask tokens. Defaults to 0.15.
        short_seq_prob: Probability of producing a short sequence. Defaults to 0.0.
        max_predictions_per_seq: Maximum number of mask tokens in each sentence. Defaults to None.
        seed: Seed for random number generator for reproducibility. Defaults to 1234.
        binary_head: Specifies whether the underlying dataset
            generates a pair of blocks along with a sentence_target or not.
            Setting it to True assumes that the underlying dataset generates a
            label for the pair of sentences which is surfaced as
            sentence_target. Defaults to True.
    """

    def __init__(
        self,
        name,
        tokenizer,
        indexed_dataset,
        data_prefix,
        max_num_samples,
        mask_lm_prob,
        max_seq_length,
        short_seq_prob=0.0,
        seed=1234,
        binary_head=True,
        masking_style="bert",
    ):

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = mask_lm_prob
        self.max_seq_length = max_seq_length
        self.binary_head = binary_head
        self.masking_style = masking_style

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.
        self.samples_mapping = get_samples_mapping(
            self.indexed_dataset,
            data_prefix,
            None,
            max_num_samples,
            self.max_seq_length - 3,  # account for added tokens
            short_seq_prob,
            self.seed,
            self.name,
            self.binary_head,
        )

        # Vocab stuff.
        self.tokenizer = tokenizer
        self.vocab_id_list = list(tokenizer.get_vocab().values())
        self.vocab_id_to_token_dict = {v: k for k, v in tokenizer.get_vocab().items()}

        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2**32 since numpy requires the seed to be between 0 and 2**32 - 1

        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2 ** 32))
        return build_training_sample(
            self.tokenizer,
            sample,
            seq_length,
            self.max_seq_length,  # needed for padding
            self.vocab_id_list,
            self.vocab_id_to_token_dict,
            self.cls_id,
            self.sep_id,
            self.mask_id,
            self.pad_id,
            self.masked_lm_prob,
            np_rng,
            self.binary_head,
            masking_style=self.masking_style,
        )


def build_training_sample(
    tokenizer,
    sample,
    target_seq_length,
    max_seq_length,
    vocab_id_list,
    vocab_id_to_token_dict,
    cls_id,
    sep_id,
    mask_id,
    pad_id,
    masked_lm_prob,
    np_rng,
    binary_head,
    masking_style="bert",
):
    """Build training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the upper bound whereas the numpy one is exclusive.
    """

    if binary_head:
        # We assume that we have at least two sentences in the sample
        assert len(sample) > 1
    assert target_seq_length <= max_seq_length

    # Divide sample into two segments (A and B).
    if binary_head:
        tokens_a, tokens_b, is_next_random = get_a_and_b_segments(sample, np_rng)
    else:
        tokens_a = []
        for j in range(len(sample)):
            tokens_a.extend(sample[j])
        tokens_b = []
        is_next_random = False

    # Truncate to `target_sequence_length`.
    max_num_tokens = target_seq_length
    truncate_segments(tokens_a, tokens_b, len(tokens_a), len(tokens_b), max_num_tokens, np_rng)

    # Build tokens and toketypes.
    tokens, tokentypes = create_tokens_and_tokentypes(tokens_a, tokens_b, cls_id, sep_id)

    # Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    (tokens, masked_positions, masked_labels, _, _) = create_masked_lm_predictions(
        tokenizer,
        tokens,
        vocab_id_list,
        vocab_id_to_token_dict,
        masked_lm_prob,
        cls_id,
        sep_id,
        mask_id,
        max_predictions_per_seq,
        np_rng,
        masking_style=masking_style,
    )

    # Padding.
    tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np = pad_and_convert_to_numpy(
        tokens, tokentypes, masked_positions, masked_labels, pad_id, max_seq_length
    )

    train_sample = Instance(
        input_ids=DistTensorData(flow.tensor(tokens_np)),
        attention_mask=DistTensorData(flow.tensor(padding_mask_np)),
        tokentype_ids=DistTensorData(flow.tensor(tokentypes_np)),
        ns_labels=DistTensorData(
            flow.tensor(int(is_next_random), dtype=flow.long), placement_idx=-1
        ),
        lm_labels=DistTensorData(flow.tensor(labels_np), placement_idx=-1),
        loss_mask=DistTensorData(flow.tensor(loss_mask_np), placement_idx=-1),
    )

    return train_sample


def pad_and_convert_to_numpy(
    tokens, tokentypes, masked_positions, masked_labels, pad_id, max_seq_length
):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(tokentypes) == num_tokens
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)
    tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length, dtype=np.bool)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.bool)

    return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np


def get_a_and_b_segments(sample, np_rng):
    """Divide sample into a and b segments."""

    # Number of sentences in the sample.
    n_sentences = len(sample)
    # Make sure we always have two sentences.
    assert n_sentences > 1, "make sure each sample has at least two sentences."

    # First part:
    # `a_end` is how many sentences go into the `A`.
    a_end = 1
    if n_sentences >= 3:
        # Note that randin in numpy is exclusive.
        a_end = np_rng.randint(1, n_sentences)
    tokens_a = []
    for j in range(a_end):
        tokens_a.extend(sample[j])

    # Second part:
    tokens_b = []
    for j in range(a_end, n_sentences):
        tokens_b.extend(sample[j])

    # Random next:
    is_next_random = False
    if np_rng.random() < 0.5:
        is_next_random = True
        tokens_a, tokens_b = tokens_b, tokens_a

    return tokens_a, tokens_b, is_next_random


def truncate_segments(tokens_a, tokens_b, len_a, len_b, max_num_tokens, np_rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    assert len_a > 0
    if len_a + len_b <= max_num_tokens:
        return False
    while len_a + len_b > max_num_tokens:
        if len_a > len_b:
            len_a -= 1
            tokens = tokens_a
        else:
            len_b -= 1
            tokens = tokens_b
        if np_rng.random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()
    return True


def create_tokens_and_tokentypes(tokens_a, tokens_b, cls_id, sep_id):
    """Merge segments A and B, add [CLS] and [SEP] and build tokentypes."""

    tokens = []
    tokentypes = []
    # [CLS].
    tokens.append(cls_id)
    tokentypes.append(0)
    # Segment A.
    for token in tokens_a:
        tokens.append(token)
        tokentypes.append(0)
    # [SEP].
    tokens.append(sep_id)
    tokentypes.append(0)
    # Segment B.
    for token in tokens_b:
        tokens.append(token)
        tokentypes.append(1)
    if tokens_b:
        # [SEP].
        tokens.append(sep_id)
        tokentypes.append(1)

    return tokens, tokentypes
