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

"""T5 Style dataset."""

import collections

import numpy as np
import oneflow as flow

from libai.data.structures import DistTensorData, Instance

from ..data_utils import create_masked_lm_predictions, get_samples_mapping


class T5Dataset(flow.utils.data.Dataset):
    """
    Dataset containing sentences for T5 training.

    Args:
        name: Name of dataset.
        tokenizer: Tokenizer to use.
        data_prefix (str): Path to the training dataset.
        indexed_dataset: Indexed dataset to use.
        max_seq_length (int, optional): Maximum length of the sequence passing into encoder.
            All values are padded to this length. Defaults to 512.
        max_seq_length_dec (int, optional): Maximum length of the sequence passing into decoder.
            All values are padded to this length. Defaults to 128.
        mask_lm_prob (float, optional): Probability to mask tokens. Defaults to 0.15.
        max_preds_per_seq (int, optional): Maximum number of masked tokens in each sentence.
            Defaults to None.
        short_seq_prob (float, optional):
            Probability of producing a short sequence. Defaults to 0.0.
        seed (int, optional):
            Seed for random number generator for reproducibility. Defaults to 1234.
    """

    def __init__(
        self,
        name,
        tokenizer,
        indexed_dataset,
        data_prefix,
        max_num_samples,
        masked_lm_prob,
        max_seq_length,
        max_seq_length_dec,
        short_seq_prob,
        seed,
    ):
        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.max_seq_length_dec = max_seq_length_dec

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
            False,
        )

        # Vocab stuff.
        self.tokenizer = tokenizer
        tokenizer.add_tokens(
            [tokenizer._bos_token, tokenizer._eos_token, *tokenizer._additional_special_tokens]
        )
        vocab = tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        self.vocab_id_list = list(inv_vocab.keys())
        self.vocab_id_to_token_dict = inv_vocab
        self.cls_id = vocab[tokenizer._cls_token]
        self.sep_id = vocab[tokenizer._sep_token]
        self.mask_id = vocab[tokenizer._mask_token]
        self.pad_id = vocab[tokenizer._pad_token]
        self.bos_id = vocab[tokenizer._bos_token]
        self.eos_id = vocab[tokenizer._eos_token]
        self.sentinel_tokens = [vocab[x] for x in tokenizer._additional_special_tokens]
        assert len(self.sentinel_tokens) > 0

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):

        start_index, end_index, seq_length = self.samples_mapping[idx]
        sample = []
        for index in range(start_index, end_index):
            sample.append(self.indexed_dataset[index])
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        return build_training_sample(
            self.tokenizer,
            sample,
            seq_length,
            self.max_seq_length,  # needed for padding
            self.max_seq_length_dec,
            self.vocab_id_list,
            self.vocab_id_to_token_dict,
            self.cls_id,
            self.sep_id,
            self.mask_id,
            self.pad_id,
            self.masked_lm_prob,
            np_rng,
            self.bos_id,
            self.eos_id,
            self.sentinel_tokens,
        )


def build_training_sample(
    tokenizer,
    sample,
    target_seq_length,
    max_seq_length,
    max_seq_length_dec,
    vocab_id_list,
    vocab_id_to_token_dict,
    cls_id,
    sep_id,
    mask_id,
    pad_id,
    masked_lm_prob,
    np_rng,
    bos_id=None,
    eos_id=None,
    sentinel_tokens=None,
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
              the opper bound whereas the numpy one is exclusive.
        bos_id: start of decoder example id
        eos_id: end of generation id
        sentinel_tokens: unique value to be substituted for every replaced span
    """

    assert target_seq_length <= max_seq_length

    # flatten sentences into one list
    tokens = [token for sentence in sample for token in sentence]

    # Truncate to `target_sequence_length`.
    max_num_tokens = target_seq_length
    len(tokens) > max_num_tokens
    tokens = tokens[:max_num_tokens]

    # Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    (tokens, masked_positions, masked_labels, _, masked_spans) = create_masked_lm_predictions(
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
        max_ngrams=10,
        geometric_dist=True,
        masking_style="t5",
    )

    # Padding.
    (
        tokens_enc,
        tokens_dec_in,
        labels,
        enc_mask,
        dec_mask,
        enc_dec_mask,
        loss_mask,
    ) = pad_and_convert_to_numpy(
        tokens,
        masked_positions,
        masked_labels,
        pad_id,
        max_seq_length,
        max_seq_length_dec,
        masked_spans,
        bos_id,
        eos_id,
        sentinel_tokens,
    )

    sample = Instance(
        encoder_input_ids=DistTensorData(tokens_enc),
        decoder_input_ids=DistTensorData(tokens_dec_in),
        encoder_attn_mask=DistTensorData(enc_mask),
        decoder_attn_mask=DistTensorData(dec_mask),
        encoder_decoder_attn_mask=DistTensorData(enc_dec_mask),
        lm_labels=DistTensorData(labels, placement_idx=-1),
        loss_mask=DistTensorData(loss_mask, placement_idx=-1),
    )
    return sample


def pad_and_convert_to_numpy(
    tokens,
    masked_positions,
    masked_labels,
    pad_id,
    max_seq_length,
    max_seq_length_dec,
    masked_spans=None,
    bos_id=None,
    eos_id=None,
    sentinel_tokens=None,
):
    """Pad sequences and convert them to numpy."""

    sentinel_tokens = collections.deque(sentinel_tokens)
    t5_input = []
    (t5_decoder_in, t5_decoder_out) = ([bos_id], [])
    (start_index, end_index) = (0, None)
    for span in masked_spans:
        flag = sentinel_tokens.popleft()

        # Append the same tokens in decoder input and output
        t5_decoder_in.append(flag)
        t5_decoder_in.extend(span.label)
        t5_decoder_out.append(flag)
        t5_decoder_out.extend(span.label)

        end_index = span.index[0]
        t5_input.extend(tokens[start_index:end_index])
        t5_input.append(flag)

        # the next start index is the token after the last span token
        start_index = span.index[-1] + 1

    # Add <eos> token to the t5_decoder_out
    t5_decoder_out.append(eos_id)

    # Add the remaining tokens to the t5 input
    t5_input.extend(tokens[start_index:])

    # assert (len(t5_input) - len(masked_spans)) + \
    #        (len(t5_decoder_in) - (len(masked_spans) + 1)) == len(tokens)

    # Some checks.

    # Encoder-side padding mask.
    num_tokens = len(t5_input)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(masked_positions) == len(masked_labels)

    # Tokens..
    filler = [pad_id] * padding_length
    tokens_enc = np.array(t5_input + filler, dtype=np.int64)

    # Decoder-side padding mask.
    num_tokens_dec = len(t5_decoder_in)
    padding_length_dec = max_seq_length_dec - num_tokens_dec
    assert padding_length_dec >= 0
    filler_dec = [pad_id] * padding_length_dec
    tokens_dec_in = np.array(t5_decoder_in + filler_dec, dtype=np.int64)

    # Create attention masks
    enc_mask = make_attention_mask(tokens_enc, tokens_enc)
    enc_dec_mask = make_attention_mask(tokens_dec_in, tokens_enc)
    dec_mask = make_attention_mask(tokens_dec_in, tokens_dec_in)
    dec_mask = dec_mask * make_history_mask(tokens_dec_in)

    # Labels mask.
    labels = t5_decoder_out + ([-1] * padding_length_dec)
    labels = np.array(labels, dtype=np.int64)

    # Loss mask
    loss_mask = ([1] * num_tokens_dec) + ([0] * padding_length_dec)
    loss_mask = np.array(loss_mask, dtype=np.bool)

    tokens_enc = flow.tensor(tokens_enc, dtype=flow.long)
    tokens_dec_in = flow.tensor(tokens_dec_in, dtype=flow.long)
    labels = flow.tensor(labels, dtype=flow.long)
    enc_mask = flow.tensor(enc_mask, dtype=flow.bool)
    dec_mask = flow.tensor(dec_mask, dtype=flow.bool)
    enc_dec_mask = flow.tensor(enc_dec_mask, dtype=flow.bool)
    loss_mask = flow.tensor(loss_mask, dtype=flow.bool)

    return tokens_enc, tokens_dec_in, labels, enc_mask, dec_mask, enc_dec_mask, loss_mask


def make_attention_mask(source_block, target_block):
    """
    Returns a 2-dimensional (2-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[None, :] >= 1) * (source_block[:, None] >= 1)
    mask = mask.astype(np.int64)
    # (source_length, target_length)
    return mask


def make_history_mask(block):
    length = block.shape[0]
    arange = np.arange(length)
    history_mask = (
        arange[
            None,
        ]
        <= arange[:, None]
    )
    history_mask = history_mask.astype(np.int64)
    return history_mask
