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
"""dataset for bert."""

import collections
import math

import numpy as np
import oneflow as flow

from libai.data.data_utils import SentenceIndexedDataset
from libai.data.structures import DistTensorData, Instance

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


class BertDataset(flow.utils.data.Dataset):
    """
    Dataset containing sentence pairs for BERT training.
    Each index corresponds to a randomly generated sentence pair.
    """

    def __init__(
        self,
        tokenizer,
        data_prefix,
        indexed_dataset,
        max_seq_length=512,
        mask_lm_prob=0.15,
        short_seq_prob=0.0,
        max_preds_per_seq=None,
        seed=1234,
        binary_head=True,
    ):
        self.seed = seed
        self.mask_lm_prob = mask_lm_prob
        self.max_seq_length = max_seq_length
        self.short_seq_prob = short_seq_prob
        self.binary_head = binary_head
        if max_preds_per_seq is None:
            max_preds_per_seq = math.ceil(max_seq_length * mask_lm_prob / 10) * 10
        self.max_preds_per_seq = max_preds_per_seq

        self.has_align_dataset = (
            isinstance(indexed_dataset, (list, tuple)) and len(indexed_dataset) > 1
        )

        self.dataset = SentenceIndexedDataset(
            data_prefix,
            indexed_dataset,
            max_seq_length=self.max_seq_length - 3,
            short_seq_prob=self.short_seq_prob,
            binary_head=self.binary_head,
        )

        self.tokenizer = tokenizer
        self.vocab_id_list = list(tokenizer.get_vocab().values())
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id

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

        if self.binary_head:
            tokens_a, tokens_b, is_next_random = self.create_random_sentence_pair(
                sents, np_rng, align_labels=align_labels
            )
        else:
            tokens_a = []
            for j in range(len(sents)):
                tokens_a.extend(sents[j])
            tokens_b = []
            is_next_random = False
            if align_labels is not None:
                align_labels_a = [label for labels in align_labels for label in labels]
                align_labels_b = []
                tokens_a = (tokens_a, align_labels_a)
                tokens_b = (tokens_b, align_labels_b)

        tokens_a, tokens_b = self.truncate_seq_pair(
            tokens_a, tokens_b, self.max_seq_length - 3, np_rng
        )

        tokens, token_types, align_labels = self.create_tokens_and_token_types(tokens_a, tokens_b)

        tokens, masked_positions, masked_labels = self.create_masked_lm_predictions(
            tokens, np_rng, token_boundary=align_labels
        )

        (tokens, token_types, labels, padding_mask, loss_mask,) = self.pad_and_convert_to_tensor(
            tokens, token_types, masked_positions, masked_labels
        )

        sample = Instance(
            tokens=DistTensorData(tokens),
            padding_mask=DistTensorData(padding_mask),
            tokentype_ids=DistTensorData(token_types),
            ns_labels=DistTensorData(
                flow.tensor(int(is_next_random), dtype=flow.long), placement_idx=-1
            ),
            lm_labels=DistTensorData(labels, placement_idx=-1),
            loss_mask=DistTensorData(loss_mask, placement_idx=-1),
        )
        return sample

    def create_random_sentence_pair(self, sample, np_rng, align_labels=None):
        num_sentences = len(sample)
        assert num_sentences > 1, "make sure each sample has at least two sentences."

        a_end = 1
        if num_sentences >= 3:
            a_end = np_rng.randint(1, num_sentences)
        tokens_a = []
        for j in range(a_end):
            tokens_a.extend(sample[j])

        tokens_b = []

        for j in range(a_end, num_sentences):
            tokens_b.extend(sample[j])

        if align_labels is not None:
            align_labels_a = []
            align_labels_b = []
            for j in range(a_end):
                align_labels_a.extend(align_labels[j])
            for j in range(a_end, num_sentences):
                align_labels_b.extend(align_labels[j])

            tokens_a = (tokens_a, align_labels_a)
            tokens_b = (tokens_b, align_labels_b)

        is_next_random = False
        if np_rng.random() < 0.5:
            is_next_random = True
            tokens_a, tokens_b = tokens_b, tokens_a

        return tokens_a, tokens_b, is_next_random

    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens, np_rng):
        """truncate sequence pair to a maximum sequence length"""
        if self.has_align_dataset:
            tokens_a, align_labels_a = tokens_a
            tokens_b, align_labels_b = tokens_b
        else:
            align_labels_a, align_labels_b = [], []

        len_a, len_b = len(tokens_a), len(tokens_b)
        while True:
            total_length = len_a + len_b
            if total_length <= max_num_tokens:
                break
            if len_a > len_b:
                trunc_tokens = tokens_a
                trunc_labels = align_labels_a
                len_a -= 1
            else:
                trunc_tokens = tokens_b
                trunc_labels = align_labels_b
                len_b -= 1

            if np_rng.random() < 0.5:
                trunc_tokens.pop(0)  # remove the first element
                if len(trunc_labels) > 0:
                    trunc_labels.pop(0)
            else:
                trunc_tokens.pop()  # remove the last element
                if len(trunc_labels) > 0:
                    trunc_labels.pop()

        if self.has_align_dataset:
            tokens_a = (tokens_a, align_labels_a)
            tokens_b = (tokens_b, align_labels_b)

        return tokens_a, tokens_b

    def create_tokens_and_token_types(self, tokens_a, tokens_b):
        """merge segments A and B, add [CLS] and [SEP] and build token types."""
        if self.has_align_dataset:
            tokens_a, align_labels_a = tokens_a
            tokens_b, align_labels_b = tokens_b

        tokens = [self.cls_id] + tokens_a + [self.sep_id]
        token_types = [0] * (len(tokens_a) + 2)
        if len(tokens_b) > 0:
            tokens = tokens + tokens_b + [self.sep_id]
            token_types = token_types + [1] * (len(tokens_b) + 1)

        if self.has_align_dataset:
            align_labels = [1] + align_labels_a + [1]
            if len(align_labels_b) > 0:
                align_labels = align_labels + align_labels_b + [1]
        else:
            align_labels = None

        return tokens, token_types, align_labels

    def mask_token(self, idx, tokens, np_rng):
        """
        helper function to mask `idx` token from `tokens` according to
        section 3.3.1 of https://arxiv.org/pdf/1810.04805.pdf
        """
        label = tokens[idx]
        if np_rng.random() < 0.8:
            new_label = self.mask_id
        else:
            if np_rng.random() < 0.5:
                new_label = label
            else:
                new_label = np_rng.choice(self.vocab_id_list)

        tokens[idx] = new_label

        return label

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

        output_tokens = list(tokens)

        if self.mask_lm_prob == 0:
            return output_tokens, masked_positions, masked_labels

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == self.cls_id or token == self.sep_id:
                continue
            # Whole Word Masking means that if we mask all of the wordpieces
            # corresponding to an original word.
            #
            # Note that Whole Word Masking does *not* change the training code
            # at all -- we still predict each WordPiece independently, softmaxed
            # over the entire vocabulary.
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
                label = self.mask_token(index, output_tokens, np_rng)
                masked_lms.append(MaskedLmInstance(index=index, label=label))

        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        for p in masked_lms:
            masked_positions.append(p.index)
            masked_labels.append(p.label)

        return output_tokens, masked_positions, masked_labels

    def pad_and_convert_to_tensor(self, tokens, token_types, masked_positions, masked_labels):
        """pad sequences and convert them to tensor"""

        # check
        num_tokens = len(tokens)
        num_pad = self.max_seq_length - num_tokens
        assert num_pad >= 0
        assert len(token_types) == num_tokens
        assert len(masked_positions) == len(masked_labels)

        # tokens and token types
        filler = [self.pad_id] * num_pad
        tokens = flow.tensor(tokens + filler, dtype=flow.long)
        token_types = flow.tensor(token_types + filler, dtype=flow.long)

        # padding mask
        padding_mask = flow.tensor([1] * num_tokens + [0] * num_pad, dtype=flow.long)

        # labels and loss mask
        labels = [-1] * self.max_seq_length
        loss_mask = [0] * self.max_seq_length
        for idx, label in zip(masked_positions, masked_labels):
            assert idx < num_tokens
            labels[idx] = label
            loss_mask[idx] = 1
        labels = flow.tensor(labels, dtype=flow.long)
        loss_mask = flow.tensor(loss_mask, dtype=flow.long)

        return tokens, token_types, labels, padding_mask, loss_mask

    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch

    def prefetch(self, indices):
        self.dataset.prefetch(indices)
