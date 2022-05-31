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

"""Roberta Style dataset."""

import numpy as np
import oneflow as flow

from libai.data.structures import DistTensorData, Instance

from ..data_utils import create_masked_lm_predictions, get_samples_mapping


class RobertaDataset(flow.utils.data.Dataset):
    """Dataset containing sentence for RoBERTa training. 
    Each index corresponds to a randomly selected sentence.

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
        masking_style="bert",
    ):
        super().__init__()

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = mask_lm_prob
        self.max_seq_length = max_seq_length
        self.masking_style = masking_style

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.
        self.samples_mapping = get_samples_mapping(
            self.indexed_dataset,
            data_prefix,
            None,
            max_num_samples,
            self.max_seq_length - 2,  # account for added tokens
            short_seq_prob,
            self.seed,
            self.name,
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
    masking_style="bert",
):
    assert target_seq_length <= max_seq_length

    tokens = []
    for j in range(len(sample)):
        tokens.extend(sample[j])
    
    max_num_tokens = target_seq_length

    # create tokens and tokentypes
    tokens, tokentypes = create_tokens_and_tokentypes(tokens, cls_id, sep_id)

    # Masking
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
        masking_style=masking_style
    )

    # Padding.
    tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np = pad_and_convert_to_numpy(
        tokens, tokentypes, masked_positions, masked_labels, pad_id, max_seq_length
    )

    train_sample = Instance(
        input_ids=DistTensorData(flow.tensor(tokens_np)),
        attention_mask=DistTensorData(flow.tensor(padding_mask_np)),
        tokentype_ids=DistTensorData(flow.tensor(tokentypes_np)),
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

    # padding mask
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length, dtype=np.int64)   

    # labels and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np



def create_tokens_and_tokentypes(tokens, cls_id, sep_id):
    # [CLS].
    tokens.insert(0, cls_id)
    # [SPE].
    tokens.append(sep_id)
    tokentypes = [0] * len(tokens)
    
    return tokens, tokentypes