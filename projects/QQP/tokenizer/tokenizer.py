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

import logging
import re

import jieba

from libai.tokenizer.tokenization_base import PreTrainedTokenizer

from .bert_tokenization import FullTokenizer as FullBertTokenizer

logger = logging.getLogger("libai." + __name__)


class _BertCNWWMTokenizer(PreTrainedTokenizer):
    """Chinese whole word BERT tokenizer."""

    def __init__(self, vocab_file, lower_case=True, vocab_extra_ids=0):
        if lower_case:
            name = "BERT Lower Case"
        else:
            name = "BERT Upper Case"
        super().__init__(name)
        self.tokenizer = FullBertTokenizer(vocab_file, do_lower_case=lower_case)
        self.cls_id = self.tokenizer.vocab["[CLS]"]
        self.sep_id = self.tokenizer.vocab["[SEP]"]
        self.pad_id = self.tokenizer.vocab["[PAD]"]
        self.mask_id = self.tokenizer.vocab["[MASK]"]
        self.unk_id = self.tokenizer.vocab["[UNK]"]
        self._additional_special_tokens = []

        # (dsachan) Add BOS and EOS tokens
        # SPECIAL_TOKENS = {"eos_token": "[EOS]", "bos_token": "[BOS]"}
        self._bos_token = "[BOS]"
        self.add_token(self._bos_token)
        self._bos_token_id = self.vocab.get(self._bos_token)

        self._eos_token = "[EOS]"
        self.add_token(self._eos_token)
        self._eos_token_id = self.vocab.get(self._eos_token)

        # (dsachan) Add additional special tokens
        # These can be used as sentinel tokens in T5 model inputs
        additional_special_tokens = []
        additional_special_tokens.extend(
            ["<extra_id_{}>".format(i) for i in range(vocab_extra_ids)]
        )
        self.add_additional_special_tokens(additional_special_tokens)

    def add_token(self, token):
        if token not in self.vocab:
            self.inv_vocab[self.vocab_size] = token
            # self.vocab_size comes from len(vocab)
            # and it will increase as we add elements
            self.vocab[token] = self.vocab_size

    def add_additional_special_tokens(self, tokens_list):
        setattr(self, "additional_special_tokens", tokens_list)
        for value in tokens_list:
            self.add_token(value)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.inv_vocab

    def tokenize(self, text):
        text_tokens = self.tokenizer.tokenize(text)
        # 使用jieba分词
        text_tokens = get_new_segment(text_tokens)
        return self.tokenizer.convert_tokens_to_ids(text_tokens)

    def decode(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return self.tokenizer.convert_tokens_to_string(tokens)

    def decode_token_ids(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        exclude_list = ["[PAD]", "[CLS]"]
        non_pads = [t for t in tokens if t not in exclude_list]

        result = ""
        for s in non_pads:
            if s.startswith("##"):
                result += s[2:]
            else:
                result += " " + s

        return result

    @property
    def cls(self):
        return self.cls_id

    @property
    def sep(self):
        return self.sep_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def mask(self):
        return self.mask_id

    @property
    def bos_token(self):
        """Beginning of sentence token id"""
        return self._bos_token

    @property
    def eos_token(self):
        """End of sentence token id"""
        return self._eos_token

    @property
    def additional_special_tokens(self):
        """All the additional special tokens you may want to use (list of strings)."""
        return self._additional_special_tokens

    @property
    def bos_token_id(self):
        """Id of the beginning of sentence token in the vocabulary."""
        return self._bos_token_id

    @property
    def eos_token_id(self):
        """Id of the end of sentence token in the vocabulary."""
        return self._eos_token_id

    @property
    def additional_special_tokens_ids(self):
        """Ids of all the additional special tokens in the vocabulary (list of integers)."""
        return [self.vocab.get(token) for token in self._additional_special_tokens]

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value


def get_new_segment(segment):
    seq_cws = jieba.cut("".join(segment) if isinstance(segment, list) else segment)
    seq_cws_dict = {x: 1 for x in seq_cws}
    new_segment = []
    i = 0
    while i < len(segment):
        if len(re.findall("[\u4E00-\u9FA5]", segment[i])) == 0:
            new_segment.append(segment[i])
            i += 1
            continue

        has_add = False
        for length in range(3, 0, -1):
            if i + length > len(segment):
                continue
            if "".join(segment[i : i + length]) in seq_cws_dict:
                new_segment.append(segment[i])
                for l in range(1, length):
                    new_segment.append("##" + segment[i + l])
                i += length
                has_add = True
                break
        if not has_add:
            new_segment.append(segment[i])
            i += 1
    return new_segment
