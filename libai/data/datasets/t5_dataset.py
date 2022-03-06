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
"""dataset for t5."""

import collections
import math

import numpy as np
import oneflow as flow

from libai.data.data_utils import SentenceIndexedDataset
from libai.data.structures import DistTensorData, Instance

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def is_start_piece(piece):
    """Check if the current word piece is the starting piece (BERT)."""
    # When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    return not piece.startswith("##")


class T5Dataset(flow.utils.data.Dataset):
    """
    Dataset containing sentences for T5 training.
    """

    def __init__(
        self,
        tokenizer,
        data_prefix,
        indexed_dataset,
        max_seq_length=512,
        max_seq_length_dec=128,
        mask_lm_prob=0.15,
        max_preds_per_seq=None,
        short_seq_prob=0.0,
        seed=1234,
    ):
        self.seed = seed
        self.mask_lm_prob = mask_lm_prob
        self.max_seq_length = max_seq_length
        self.max_seq_length_dec = max_seq_length_dec
        self.short_seq_prob = short_seq_prob
        if max_preds_per_seq is None:
            max_preds_per_seq = math.ceil(max_seq_length * mask_lm_prob / 10) * 10
        self.max_preds_per_seq = max_preds_per_seq

        self.dataset = SentenceIndexedDataset(
            data_prefix,
            indexed_dataset,
            max_seq_length=self.max_seq_length - 2,
            short_seq_prob=self.short_seq_prob,
        )

        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.special_tokens = tokenizer.additional_special_tokens_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sents = self.dataset[idx]

        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2 ** 32 since numpy requires the seed to be between 0 and 2 ** 32 - 1
        np_rng = np.random.RandomState(seed=(self.seed + idx))

        tokens = [token for sent in sents for token in sent]
        tokens = tokens[: self.max_seq_length - 2]

        (
            tokens,
            masked_positions,
            masked_labels,
            masked_spans,
        ) = self.create_masked_lm_predictions(tokens, np_rng, geometric_dist=True, max_ngrams=10)

        (
            encoder_input,
            decoder_input,
            labels,
            encoder_padding_mask,
            decoder_padding_mask,
            encoder_decoder_padding_mask,
            loss_mask,
        ) = self.pad_and_convert_to_numpy(tokens, masked_spans)

        sample = Instance(
            encoder_input_ids=DistTensorData(encoder_input),
            decoder_input_ids=DistTensorData(decoder_input),
            encoder_attn_mask=DistTensorData(encoder_padding_mask),
            decoder_attn_mask=DistTensorData(decoder_padding_mask),
            encoder_decoder_attn_mask=DistTensorData(encoder_decoder_padding_mask),
            lm_labels=DistTensorData(labels, placement_idx=-1),
            loss_mask=DistTensorData(loss_mask, placement_idx=-1),
        )
        return sample

    def create_masked_lm_predictions(
        self,
        tokens,
        np_rng,
        max_ngrams=3,
        do_whole_word_mask=True,
        favor_longer_ngram=False,
        geometric_dist=False,
    ):
        """Creates the predictions for the masked LM objective.
        Note: Tokens here are vocab ids and not text tokens."""

        cand_indexes = []
        token_boundary = [0] * len(tokens)
        new_tokens = []

        for (i, token) in enumerate(tokens):
            new_tokens.append(token % len(self.tokenizer))

            if token == self.cls_id or token == self.sep_id:
                token_boundary[i] = 1
                continue
            # Whole Word Masking means that if we mask all of the wordpieces
            # corresponding to an original word.
            #
            # Note that Whole Word Masking does *not* change the training code
            # at all -- we still predict each WordPiece independently, softmaxed
            # over the entire vocabulary.
            if (
                do_whole_word_mask
                and len(cand_indexes) >= 1
                and not is_start_piece(self.tokenizer._convert_id_to_token(token))
            ):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
                if is_start_piece(self.tokenizer._convert_id_to_token(token)):
                    token_boundary[i] = 1

        tokens = new_tokens

        masked_positions = []
        masked_labels = []
        masked_spans = []

        output_tokens = list(tokens)

        if self.mask_lm_prob == 0:
            return output_tokens, masked_positions, masked_labels, masked_spans

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if do_whole_word_mask and len(cand_indexes) >= 1 and token_boundary[i] == 0:
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        num_to_predict = min(
            self.max_preds_per_seq, max(1, int(round(len(tokens) * self.mask_lm_prob)))
        )

        ngrams = np.arange(1, max_ngrams + 1, dtype=np.int64)
        if not geometric_dist:
            # By default, we set the probilities to favor shorter ngram sequences.
            pvals = 1.0 / np.arange(1, max_ngrams + 1)
            pvals /= pvals.sum(keepdims=True)
            if favor_longer_ngram:
                pvals = pvals[::-1]

        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx : idx + n])
            ngram_indexes.append(ngram_index)

        np_rng.shuffle(ngram_indexes)

        masked_lms = []
        covered_indexes = set()
        for cand_index_set in ngram_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes:
                        continue

            if not geometric_dist:
                n = np_rng.choice(
                    ngrams[: len(cand_index_set)],
                    p=pvals[: len(cand_index_set)]
                    / pvals[: len(cand_index_set)].sum(keepdims=True),
                )
            else:
                # Sampling "n" from the geometric distribution and clipping it to
                # the max_ngrams. Using p=0.2 default from the SpanBERT paper
                # https://arxiv.org/pdf/1907.10529.pdf (Sec 3.1)
                n = min(np_rng.geometric(0.2), max_ngrams)

            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
            # Repeatedly looking for a candidate that does not exceed the
            # maximum number of predictions by trying shorter ngrams.
            while len(masked_lms) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

            masked_spans.append(
                MaskedLmInstance(index=index_set, label=[tokens[index] for index in index_set])
            )

        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        masked_spans = sorted(masked_spans, key=lambda x: x.index[0])

        for p in masked_lms:
            masked_positions.append(p.index)
            masked_labels.append(p.label)

        return output_tokens, masked_positions, masked_labels, masked_spans

    def pad_and_convert_to_numpy(self, tokens, masked_spans):
        """pad sequences and convert them to numpy array"""

        special_tokens = collections.deque(self.special_tokens)
        encoder_input, decoder_input, decoder_output = [], [], []

        decoder_input.append(self.bos_id)
        start_index, end_index = 0, None

        for span in masked_spans:
            flag = special_tokens.popleft()

            decoder_input.append(flag)
            decoder_input.extend(span.label)
            decoder_output.append(flag)
            decoder_output.extend(span.label)

            end_index = span.index[0]
            encoder_input.extend(tokens[start_index:end_index])
            encoder_input.append(flag)

            start_index = span.index[-1] + 1

        decoder_output.append(self.eos_id)
        encoder_input.extend(tokens[start_index:])

        # check
        num_tokens = len(encoder_input)
        num_pad = self.max_seq_length - num_tokens
        assert num_pad >= 0

        filler = [self.pad_id] * num_pad
        encoder_input = np.array(encoder_input + filler, dtype=np.int64)

        num_tokens_dec = len(decoder_input)
        num_pad_dec = self.max_seq_length_dec - num_tokens_dec
        assert num_pad_dec >= 0

        # tokens and token types
        filler_dec = [self.pad_id] * num_pad_dec
        decoder_input = np.array(decoder_input + filler_dec, dtype=np.int64)

        # Create attention masks
        encoder_padding_mask = self.make_attention_mask(encoder_input, encoder_input)
        decoder_padding_mask = self.make_attention_mask(decoder_input, decoder_input)
        encoder_decoder_padding_mask = self.make_attention_mask(decoder_input, encoder_input)
        decoder_padding_mask = decoder_padding_mask * self.make_history_mask(decoder_input)
        
        # Labels mask.
        labels = decoder_output + ([-1] * num_pad_dec)
        labels = np.array(labels, dtype=np.int64)

        # Loss mask
        loss_mask = ([1] * num_tokens_dec) + ([0] * num_pad_dec)
        loss_mask = np.array(loss_mask, dtype=np.int64)
        
        encoder_input = flow.tensor(encoder_input, dtype=flow.long)
        decoder_input = flow.tensor(decoder_input, dtype=flow.long)
        labels = flow.tensor(labels, dtype=flow.long)
        encoder_padding_mask = flow.tensor(encoder_padding_mask, dtype=flow.long)
        decoder_padding_mask = flow.tensor(decoder_padding_mask, dtype=flow.long)
        encoder_decoder_padding_mask = flow.tensor(encoder_decoder_padding_mask, dtype=flow.long)
        loss_mask = flow.tensor(loss_mask, dtype=flow.long)

        return (
            encoder_input,
            decoder_input,
            labels,
            encoder_padding_mask,
            decoder_padding_mask,
            encoder_decoder_padding_mask,
            loss_mask,
        )

    def make_attention_mask(self, source_block, target_block):
        """
        Returns a 2-dimensional (2-D) attention mask
        :param source_block: 1-D array
        :param target_block: 1-D array
        """
        mask = (target_block[None, :] >= 1) * (source_block[:, None] >= 1)
        mask = mask.astype(np.int64)
        # (source_length, target_length)
        return mask

    def make_history_mask(self, block):
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

    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch

    def prefetch(self, indices):
        self.dataset.prefetch(indices)
