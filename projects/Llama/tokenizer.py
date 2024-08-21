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
import sentencepiece as spm

import libai.utils.distributed as dist


class LlamaTokenizer:
    def __init__(
        self,
        pretrained_model_path,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<unk>",
        bos_token_id=None,
        eos_token_id=None,
    ):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(pretrained_model_path)

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.bos_token_id = self.sp_model.bos_id() if self.sp_model.bos_id() else bos_token_id
        self.eos_token_id = self.sp_model.eos_id() if self.sp_model.eos_id() else eos_token_id
        self.pad_token_id = 0

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    def encode(self, text):
        tokens = self.sp_model.encode(text)
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
            tokens = [self.sp_model.encode(text)[:max_length]]

        if isinstance(text, list):
            tokens = [self.sp_model.encode(s)[:max_length] for s in text]
            if padding:
                max_length = max([len(i) for i in tokens])
                tokens = [t + (max_length - len(t)) * [self.pad_token_id] for t in tokens]

        if add_bos:
            tokens = [[self.bos_token_id] + token for token in tokens]
        if add_eos:
            tokens = [token + [self.eos_token_id] for token in tokens]

        if device == "cuda":
            sbp = kwargs.get("sbp", dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]))
            placement = kwargs.get("placement", flow.placement("cuda", [0]))
            return_token_ids = flow.tensor(tokens, sbp=sbp, placement=placement, dtype=flow.long)
        else:
            return_token_ids = flow.tensor(tokens, dtype=flow.long)
        return return_token_ids

    def decode(self, tokens):
        if isinstance(tokens, flow.Tensor):
            tokens = tokens.tolist()
        return self.sp_model.decode(tokens)

    def convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    def convert_id_to_token(self, index):
        return self.sp_model.IdToPiece(index)
