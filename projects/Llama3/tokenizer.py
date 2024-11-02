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

import oneflow as flow
# import sentencepiece as spm
import tiktoken
from pathlib import Path
from tiktoken.load import load_tiktoken_bpe

import libai.utils.distributed as dist


class LlamaTokenizer:
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501
    def __init__(
        self,
        pretrained_model_path,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<unk>",
        bos_token_id=None,
        eos_token_id=None,
    ):
        mergeable_ranks = load_tiktoken_bpe(pretrained_model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<s>",
            "</s>",
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.tik_model = tiktoken.Encoding(
            name=Path(pretrained_model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.bos_token_id = self.special_tokens["<s>"]
        self.eos_token_id = self.special_tokens["</s>"]
        self.pad_token_id = 0

    @property
    def vocab_size(self):
        return self.tik_model.n_vocab

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    def encode(self, text):
        tokens = self.tik_model.encode(text)
        return tokens

    def tokenize(
        self,
        text,
        add_bos=False,
        add_eos=False,
        padding=False,
        device="cuda",
        max_length=4096,
        **kwargs
    ):
        if isinstance(text, str):
            tokens = [self.tik_model.encode(text)[:max_length]]

        if isinstance(text, list):
            tokens = [self.tik_model.encode(s)[:max_length] for s in text]
            if padding:
                max_length = max([len(i) for i in tokens])
                tokens = [t + (max_length - len(t)) * [self.pad_token_id] for t in tokens]

        if add_bos:
            tokens = [[self.bos_token_id] + token for token in tokens]
        if add_eos:
            tokens = [token + [self.eos_token_id] for token in tokens]

        if device:
            sbp = kwargs.get("sbp", dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]))
            placement = kwargs.get("placement", flow.placement(device, [0]))
            return_token_ids = flow.tensor(tokens, sbp=sbp, placement=placement, dtype=flow.long)
        else:
            return_token_ids = flow.tensor(tokens, dtype=flow.long)
        return return_token_ids

    def decode(self, tokens):
        if isinstance(tokens, flow.Tensor):
            tokens = tokens.tolist()
        return self.tik_model.decode(tokens)

    def convert_token_to_id(self, token):
        return self.tik_model.encode_single_token(token)

    def convert_id_to_token(self, index):
        return self.tik_model.decode_single_token_bytes(index)
