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

import json
import os
import unittest

from libai.tokenizer.tokenization_roberta import VOCAB_FILES_NAMES, RobertaTokenizer
from tests.tokenizer.test_tokenization_common import TokenizerTesterMixin


class RobertaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = RobertaTokenizer

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return RobertaTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = RobertaTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "lower newer"
        bpe_tokens = ["l", "o", "w", "er", "\u0120", "n", "e", "w", "er"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [0, 1, 2, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def roberta_dict_integration_testing(self):
        tokenizer = self.get_tokenizer()

        self.assertListEqual(
            tokenizer.encode("Hello world!", add_special_tokens=False), [0, 31414, 232, 328, 2]
        )
        self.assertListEqual(
            tokenizer.encode("Hello world! cécé herlolip 418", add_special_tokens=False),
            [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2],
        )


if __name__ == "__main__":
    unittest.main()
