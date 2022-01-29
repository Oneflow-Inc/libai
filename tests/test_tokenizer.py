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

import unittest
from unittest import TestCase

from omegaconf import DictConfig, OmegaConf

from libai.config import LazyCall
from libai.tokenizer import BertTokenizer, build_tokenizer


class TestTokenizer(TestCase):
    def test_tokenizer_build_with_register(self):
        # build_tokenizer for register
        token_cfg = dict(
            tokenizer=dict(
                name="BertTokenizer",
                vocab_file="tests/bert-base-chinese-vocab.txt",
                do_lower_case=True,
                additional_special_tokens=[
                    "<special_id_0>",
                    "<special_id_1>",
                    "<special_id_2>",
                    "<special_id_3>",
                ],
                # do_chinese_wwm=True,
            ),
            append_eod=False,
            make_vocab_size_divisible_by=128,
        )

        register_cfg = DictConfig(token_cfg)

        tokenizer = build_tokenizer(register_cfg)
        self.assertTrue(len(tokenizer) == 21128)

    def test_tokenizer_build_with_lazy(self):
        lazy_cfg = OmegaConf.create()
        lazy_cfg.tokenizer = LazyCall(BertTokenizer)(
            vocab_file="tests/bert-base-chinese-vocab.txt",
            do_lower_case=True,
        )
        lazy_cfg.append_eod = False
        lazy_cfg.make_vocab_size_divisible_by = 1

        tokenizer = build_tokenizer(lazy_cfg)
        self.assertTrue(len(tokenizer) == 21128)

        inputs = "今天天气真不错。"
        print(tokenizer.tokenize(inputs))

        tokens = tokenizer.encode(inputs)
        print(tokens)

        print(tokenizer.decode(tokens))

        print(tokenizer.vocab_size)
        print(len(tokenizer))
        print(tokenizer.additional_special_tokens)
        print(tokenizer.added_tokens_encoder)


if __name__ == "__main__":
    unittest.main()
