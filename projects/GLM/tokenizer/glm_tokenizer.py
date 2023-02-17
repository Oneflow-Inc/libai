# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
# Copyright 2022 The HuggingFace Inc. team.
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
import os
from shutil import copyfile
from typing import List, Optional, Tuple

import oneflow as flow
import sentencepiece as spm

from libai.tokenizer import BertTokenizer, GPT2Tokenizer, PreTrainedTokenizer, RobertaTokenizer

logger = logging.getLogger(__name__)


class GLMTokenizerMixin(PreTrainedTokenizer):
    @property
    def sop_token(self) -> Optional[str]:
        return "<|startofpiece|>"

    @property
    def sop_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the start token in the vocabulary, used when training a model with
        autoregressive blank filling.
        """
        return self.convert_tokens_to_ids(self.sop_token)

    @property
    def eop_token(self) -> Optional[str]:
        return "<|endofpiece|>"

    @property
    def eop_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the end token in the vocabulary, used when training a model with
        autoregressive blank filling.
        """
        return self.convert_tokens_to_ids(self.eop_token)

    @property
    def gmask_token_id(self) -> int:
        return self.convert_tokens_to_ids("[gMASK]")

    @property
    def smask_token_id(self) -> int:
        return self.convert_tokens_to_ids("[sMASK]")

    @property
    def mask_token_ids(self):
        return [self.mask_token_id, self.smask_token_id, self.gmask_token_id]

    def _build_input_for_multiple_choice(self, context, choices):
        context_id = context["input_ids"]
        if flow.is_tensor(context_id):
            context_id = context_id.tolist()

        division = len(context_id)
        mask_position = context_id.index(self.mask_token_id)

        token = flow.tensor(context_id, dtype=flow.long)
        attention_mask = [context["attention_mask"].expand(division, -1)]
        position_id = flow.arange(division, dtype=flow.long)
        block_position_id = flow.zeros(division, dtype=flow.long)

        choice_ids, choice_indices = [], []

        for choice_str in choices:
            res = self.encode(choice_str)
            choice = flow.tensor(res, dtype=flow.long)
            choice_ids.append(choice)
            choice_indices.append(
                flow.arange(len(token), len(token) + len(choice), dtype=flow.long)
            )
            attention_mask.append(flow.tril(flow.ones((len(choice), len(choice)), dtype=flow.long)))

            token = flow.cat(
                (token, flow.tensor([self.sop_token_id], dtype=flow.long), choice[:-1])
            )
            position_id = flow.cat(
                (position_id, flow.tensor([mask_position] * len(choice), dtype=flow.long))
            )
            block_position_id = flow.cat(
                (block_position_id, flow.arange(1, 1 + len(choice), dtype=flow.long))
            )

        attention_mask = flow.block_diag(*attention_mask)
        attention_mask[division:, :division] = context["attention_mask"].unsqueeze(0)

        return {
            "input_ids": token,
            "position_ids": flow.stack((position_id, block_position_id)),
            "attention_mask": attention_mask,
            "choice_ids": choice_ids,
            "choice_indices": choice_indices,
        }

    def _pad_batch(self, tokens, position_ids, attention_mask, max_seq_length):
        pad_length = max_seq_length - len(tokens)
        attention_mask = flow.nn.functional.pad(
            attention_mask,
            (0, pad_length, 0, pad_length),
            mode="constant",
            value=0,
        )
        tokens = flow.cat((tokens, flow.zeros(pad_length, dtype=flow.long)))
        position_ids = flow.cat(
            (position_ids, position_ids[..., -1:].expand(-1, pad_length)), dim=-1
        )
        return tokens, position_ids, attention_mask

    def _collate(self, samples):
        TILE = 1
        length_to_pad = (
            (max(map(lambda spl: len(spl["input_ids"]), samples)) + TILE - 1) // TILE * TILE
        )

        token_batch, position_id_batch, attention_mask_batch = [], [], []
        choices_batch, choice_target_ids_batch = [], []

        for sample in samples:
            token, position_id, attention_mask = self._pad_batch(
                sample["input_ids"], sample["position_ids"], sample["attention_mask"], length_to_pad
            )
            token_batch.append(token)
            position_id_batch.append(position_id)
            attention_mask_batch.append(attention_mask)
            choices_batch.append(sample["choice_ids"])
            choice_target_ids_batch.append(sample["choice_indices"])
        return {
            "input_ids": flow.stack(token_batch),
            "position_ids": flow.stack(position_id_batch),
            "attention_mask": flow.stack(attention_mask_batch).unsqueeze(1),
            "choice_ids": choices_batch,
            "choice_indices": choice_target_ids_batch,
        }

    def build_inputs_for_multiple_choice(self, model_input, choices, max_length=None):
        samples = [
            {key: value[i] for key, value in model_input.items()}
            for i in range(len(model_input["input_ids"]))
        ]
        samples = [
            self._build_input_for_multiple_choice(sample, choice)
            for sample, choice in zip(samples, choices)
        ]
        inputs = self._collate(samples)
        return inputs

    def build_inputs_for_generation(
        self, model_input, max_gen_length=512, targets=None, padding=False
    ):
        mask_ids = self.mask_token_ids
        input_ids = model_input["input_ids"]
        batch_size, seq_length = input_ids.shape[:2]
        position_id, block_position_id = list(range(seq_length)), [0 for _ in range(seq_length)]
        position_ids, block_position_ids = [], []
        labels = None
        if targets is not None:
            is_batched = isinstance(targets, (list, tuple))
            targets = self.encode(targets)
            if not is_batched:
                targets = [targets]
            assert len(targets) == len(input_ids)
            targets = [(target + [self.eop_token_id])[:max_gen_length] for target in targets]
            if not padding:
                max_gen_length = max(map(len, targets))
            targets = [[self.sop_token_id] + target for target in targets]
            labels = [target[1:] for target in targets]
            targets = [
                target + [self.pad_token_id] * (max_gen_length + 1 - len(target))
                for target in targets
            ]
            labels = [label + [-100] * (max_gen_length - len(label)) for label in labels]
            targets = flow.tensor(targets, dtype=input_ids.dtype)
            labels = flow.tensor(labels, dtype=input_ids.dtype)
            labels = flow.cat((input_ids.new_full((batch_size, seq_length), -100), labels), dim=1)
        for i in range(batch_size):
            mask_positions = []
            for mask_id in mask_ids:
                mask_positions += (input_ids[i] == mask_id).nonzero(as_tuple=True)[0].tolist()
            if not mask_positions:
                raise ValueError("Cannot find mask token in the input")
            mask_positions.sort()
            mask_pos = mask_positions[0]
            position_ids.append(position_id + [mask_pos] * max_gen_length)
            block_position_ids.append(block_position_id + list(range(1, max_gen_length + 1)))
        position_ids = flow.tensor(position_ids, dtype=input_ids.dtype)
        block_position_ids = flow.tensor(block_position_ids, dtype=input_ids.dtype)
        position_ids = flow.stack((position_ids, block_position_ids), dim=1)
        attention_mask = model_input["attention_mask"]
        attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_length + max_gen_length, -1)
        generation_attention_mask = (
            flow.cat(
                [
                    attention_mask.new_zeros((seq_length, max_gen_length)),
                    flow.tril(attention_mask.new_ones((max_gen_length, max_gen_length))),
                ],
                dim=0,
            )
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        attention_mask = flow.cat((attention_mask, generation_attention_mask), dim=2)
        attention_mask = attention_mask.unsqueeze(1)
        if targets is None:
            input_ids = flow.cat(
                (input_ids, input_ids.new_full((batch_size, 1), self.sop_token_id)), dim=-1
            )
        else:
            input_ids = flow.cat((input_ids, targets[:, :-1]), dim=1)
        batch = {"input_ids": input_ids, "position_ids": position_ids}
        if labels is None:
            batch["generation_attention_mask"] = attention_mask
        else:
            batch["attention_mask"] = attention_mask
            batch["labels"] = labels
        return batch


class GLMRobertaTokenizer(RobertaTokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    truncation_side: str = "left"

    @property
    def gmask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support gMASK")

    @property
    def smask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support sMASK")

    @property
    def mask_token_ids(self):
        return [self.mask_token_id]


class GLMChineseTokenzier(GLMTokenizerMixin):
    vocab_files_names = {"vocab_file": "cog-pretrain.model"}
    truncation_side: str = "left"

    def __init__(
        self,
        vocab_file,
        eos_token="<|endoftext|>",
        unk_token="[UNK]",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|startofpiece|>", "<|endofpiece|>", "[gMASK]", "[sMASK]"],
        add_bos_token=False,
        **kwargs,
    ):
        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.add_bos_token = add_bos_token
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        self._eos_token = "<|endoftext|>"
        self._unk_token = "[UNK]"
        self._pad_token = "<|endoftext|>"
        self._cls_token = "[CLS]"
        self._mask_token = "[MASK]"

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        return self.sp_model.decode(tokens)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + self.vocab_files_names["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(
            self.vocab_file
        ):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the
            appropriate special tokens.
        """
        assert token_ids_1 is None
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]
        return cls + token_ids_0 + eos


class GLMGPT2Tokenizer(GPT2Tokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    truncation_side: str = "left"

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_bos_token=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            errors,
            unk_token,
            bos_token,
            eos_token,
            add_bos_token,
            **kwargs,
        )
        self.cls_token = "[CLS]"
        self.mask_token = "[MASK]"

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the
            appropriate special tokens.
        """
        assert token_ids_1 is None
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]
        return cls + token_ids_0 + eos


class GLMBertTokenizer(BertTokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    truncation_side: str = "left"

    @property
    def gmask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support gMASK")

    @property
    def smask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support sMASK")

    @property
    def mask_token_ids(self):
        return [self.mask_token_id]
