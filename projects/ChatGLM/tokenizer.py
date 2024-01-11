# --------------------------------------------------------
# Refer to code from:
# https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py
# --------------------------------------------------------

import json
import os
import re
from enum import Enum
from typing import List, Optional

from sentencepiece import SentencePieceProcessor

from libai.tokenizer.tokenization_base import PreTrainedTokenizer

EncodedInput = List[int]


class PaddingStrategy(str, Enum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`].
    Useful for tab-completion in an IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"


class SPTokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        role_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"] + role_special_tokens
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1
        self.role_special_token_expression = "|".join(
            [re.escape(token) for token in role_special_tokens]
        )

    def tokenize(self, s: str, encode_special_tokens=False):
        if encode_special_tokens:
            last_index = 0
            t = []
            for match in re.finditer(self.role_special_token_expression, s):
                if last_index < match.start():
                    t.extend(self.sp_model.EncodeAsPieces(s[last_index : match.start()]))
                t.append(s[match.start() : match.end()])
                last_index = match.end()
            if last_index < len(s):
                t.extend(self.sp_model.EncodeAsPieces(s[last_index:]))
            return t
        else:
            return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        text, buffer = "", []
        for token in t:
            if token in self.index_special_tokens:
                if buffer:
                    text += self.sp_model.decode(buffer)
                    buffer = []
                text += self.index_special_tokens[token]
            else:
                buffer.append(token)
        if buffer:
            text += self.sp_model.decode(buffer)
        return text

    def decode_tokens(self, tokens: List[str]) -> str:
        text = self.sp_model.DecodePieces(tokens)
        return text

    def convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.index_special_tokens:
            return self.index_special_tokens[index]
        if (
            index in [self.eos_id, self.bos_id, self.pad_id]
            or index < 0
            or index > self.sp_model.vocab_size()
        ):
            return ""
        return self.sp_model.IdToPiece(index)


class ChatGLMTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}

    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(
        self,
        vocab_file,
        padding_side="left",
        clean_up_tokenization_spaces=False,
        encode_special_tokens=False,
        **kwargs,
    ):
        self.name = "GLMTokenizer"

        self.vocab_file = vocab_file
        self.tokenizer = SPTokenizer(vocab_file)
        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<pad>": self.tokenizer.pad_id,
        }
        self.encode_special_tokens = encode_special_tokens
        super().__init__(
            padding_side=padding_side,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            encode_special_tokens=encode_special_tokens,
            **kwargs,
        )

    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert (
            token in self.tokenizer.special_tokens
        ), f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]

    @property
    def unk_token(self) -> str:
        return "<unk>"

    @property
    def pad_token(self) -> str:
        return "<unk>"

    @property
    def pad_token_id(self):
        return self.get_command("<pad>")

    @property
    def eos_token(self) -> str:
        return "</s>"

    @property
    def eos_token_id(self):
        return self.get_command("<eos>")

    @property
    def vocab_size(self):
        return self.tokenizer.n_words

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text, encode_special_tokens=self.encode_special_tokens)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.tokenizer.convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.decode_tokens(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.
        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.
        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, self.vocab_files_names["vocab_file"])
        else:
            vocab_file = save_directory

        with open(self.vocab_file, "rb") as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)

    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        return prefix_tokens

    def build_single_message(self, role, metadata, message):
        assert role in ["system", "user", "assistant", "observation"], role
        role_tokens = [self.get_command(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n")
        message_tokens = self.tokenizer.encode(message)
        tokens = role_tokens + message_tokens
        return tokens

    def build_chat_input(self, query, history=None, role="user"):
        if history is None:
            history = []
        input_ids = []
        for item in history:
            content = item["content"]
            if item["role"] == "system" and "tools" in item:
                content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
            input_ids.extend(
                self.build_single_message(item["role"], item.get("metadata", ""), content)
            )
        input_ids.extend(self.build_single_message(role, "", query))
        input_ids.extend([self.get_command("<|assistant|>")])
        return [self.get_prefix_tokens() + input_ids]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence
        classification tasks by concatenating and adding special tokens.
        A BERT sequence has the following format:
        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with
            the appropriate special tokens.
        """
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
        return token_ids_0

    def _pad(
        self,
        encoded_inputs: List[List],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.LONGEST,
    ) -> dict:

        # Load from model defaults
        # assert self.padding_side == "left"

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max([len(l) for l in encoded_inputs])

        for i in range(len(encoded_inputs)):
            difference = max_length - len(encoded_inputs[i])

            encoded_inputs[i] = [self.pad_token_id] * difference + encoded_inputs[i]

        return encoded_inputs

    def encode(self, text, return_tensors=None, is_global=False, **kwargs):
        if isinstance(text, str):
            text = [text]
        tokens = [self.tokenize(t) for t in text]
        token_ids_list = self.convert_tokens_to_ids(tokens)
        token_ids_list = self._pad(token_ids_list, padding_strategy=PaddingStrategy.LONGEST)
        token_ids_list = [
            self.build_inputs_with_special_tokens(token_ids) for token_ids in token_ids_list
        ]
        token_ids_list = self.convert_to_tensors(
            token_ids_list, return_tensors=return_tensors, is_global=is_global, **kwargs
        )
        return token_ids_list
