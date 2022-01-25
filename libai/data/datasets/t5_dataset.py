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
        mask_lm_prob=0.15,
        max_preds_per_seq=None,
        short_seq_prob=0.0,
        seed=1234,
    ):
        self.seed = seed
        self.mask_lm_prob = mask_lm_prob
        self.max_seq_length = max_seq_length
        self.short_seq_prob = short_seq_prob
        if max_preds_per_seq is None:
            max_preds_per_seq = math.ceil(max_seq_length * mask_lm_prob / 10) * 10
        self.max_preds_per_seq = max_preds_per_seq

        self.has_align_dataset = (
            isinstance(indexed_dataset, (list, tuple)) and len(indexed_dataset) > 1
        )

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
        self.special_tokens = tokenizer.additional_special_tokens_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2 ** 32 since numpy requres the seed to be between 0 and 2 ** 32 - 1
        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2 ** 32))

        sents = self.dataset[idx]
        align_labels = None
        if self.has_align_dataset:
            sents, align_labels = sents

        tokens = [token for sent in sents for token in sent]
        align_labels = (
            [label for labels in align_labels for label in labels]
            if align_labels is not None
            else None
        )

        (
            tokens,
            masked_positions,
            masked_labels,
            masked_spans,
        ) = self.create_masked_lm_predictions(tokens, np_rng, token_boundary=align_labels)

        (
            encoder_input,
            decoder_input,
            labels,
            encoder_padding_mask,
            decoder_padding_mask,
            loss_mask,
        ) = self.pad_and_convert_to_numpy(tokens, masked_spans)

        sample = Instance(
            encoder_input=DistTensorData(encoder_input),
            decoder_input=DistTensorData(decoder_input),
            encoder_padding_mask=DistTensorData(encoder_padding_mask),
            decoder_padding_mask=DistTensorData(decoder_padding_mask),
            labels=DistTensorData(labels, placement_idx=-1),
            loss_mask=DistTensorData(loss_mask, placement_idx=-1),
        )
        return sample

    def create_masked_lm_predictions(
        self,
        tokens,
        np_rng,
        max_ngrams=3,
        do_whole_word_mask=False,
        token_boundary=None,
        favor_longer_ngram=False,
        geometric_dist=False,
    ):
        """Creates the predictions for the masked LM objective.
        Note: Tokens here are vocab ids and not text tokens."""

        if do_whole_word_mask:
            assert (
                token_boundary is not None
            ), "token_boundary must be privided when do_whole_word_mask is True."

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
        encoder_input = np.array(encoder_input + filler, dtype=np.long)
        encoder_input = flow.tensor(encoder_input, dtype=flow.long)

        num_tokens_dec = len(decoder_input)
        num_pad_dec = self.max_seq_length - num_tokens_dec
        assert num_pad_dec >= 0

        # tokens and token types
        filler_dec = [self.pad_id] * num_pad_dec
        decoder_input = np.array(decoder_input + filler_dec, dtype=np.long)
        decoder_input = flow.tensor(decoder_input, dtype=flow.long)

        # padding mask
        encoder_padding_mask = flow.tensor([1] * num_tokens + [0] * num_pad, dtype=flow.long)
        decoder_padding_mask = flow.tensor(
            [1] * num_tokens_dec + [0] * num_pad_dec, dtype=flow.long
        )

        # labels and loss mask
        labels = flow.tensor(decoder_output + [-1] * num_pad_dec, dtype=flow.long)
        loss_mask = [1] * num_tokens_dec + [0] * num_pad_dec
        loss_mask = flow.tensor(loss_mask, dtype=flow.long)

        return (
            encoder_input,
            decoder_input,
            labels,
            encoder_padding_mask,
            decoder_padding_mask,
            loss_mask,
        )

    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch

    def prefetch(self, indices):
        self.dataset.prefetch(indices)
