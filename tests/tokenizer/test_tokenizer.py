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

import sys

from omegaconf import DictConfig, OmegaConf

sys.path.append(".")

from libai.config import LazyCall
from libai.tokenizer import BertTokenizer, build_tokenizer

# build_tokenizer for register
default_cfg = dict(
    tokenizer=dict(
        name="BertTokenizer",
        vocab_file="bert-base-chinese-vocab.txt",
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

reg_cfg = DictConfig(default_cfg)

reg_tokenizer = build_tokenizer(reg_cfg)


lazy_cfg = OmegaConf.create()
lazy_cfg.tokenizer = LazyCall(BertTokenizer)(
    vocab_file="bert-base-chinese-vocab.txt",
    do_lower_case=True,
    # do_chinese_wwm=True,
)
lazy_cfg.append_eod = False
lazy_cfg.make_vocab_size_divisible_by = 1

lazy_tokenizer = build_tokenizer(lazy_cfg)

tokenizer = build_tokenizer(reg_cfg)
inputs = "今天天气真不错。"
print(tokenizer.tokenize(inputs))

tokens = tokenizer.encode(inputs)
print(tokens)

print(tokenizer.decode(tokens))

print(tokenizer.vocab_size)
print(len(tokenizer))
print(tokenizer.additional_special_tokens)
print(tokenizer.added_tokens_encoder)
