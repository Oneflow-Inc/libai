# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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

"""Tokenization classes for bert (wordpieces)."""

import collections
import logging
import os
import re
import unicodedata
from io import open

from .build import TOKENIZER_REGISTRY
from .tokenization_base import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/vocab.txt",
        "bert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/vocab.txt",
        "bert-large-cased": "https://huggingface.co/bert-large-cased/resolve/main/vocab.txt",
        "bert-base-chinese": "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bert-base-uncased": 512,
    "bert-large-uncased": 512,
    "bert-base-cased": 512,
    "bert-large-cased": 512,
    "bert-base-chinese": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "bert-base-uncased": {"do_lower_case": True},
    "bert-large-uncased": {"do_lower_case": True},
    "bert-base-cased": {"do_lower_case": False},
    "bert-large-cased": {"do_lower_case": False},
    "bert-base-chinese": {"do_lower_case": False},
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_chinese_substr(char):
    return re.findall("##[\u4E00-\u9FA5]", char)


@TOKENIZER_REGISTRY.register()
class BertTokenizer(PreTrainedTokenizer):
    """
    Construct a BERT tokenizer. Based on WordPiece.

    Args:
        vocab_file (:obj:`str`):
            Path to a one-wordpiece-per-line vocabulary file.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to lower case the input
            Only has an effect when do_basic_tokenize=True.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to do basic tokenization before wordpiece.
        never_split (:obj:`Iterable`, `optional`):
            List of tokens which will never be split during tokenization.
            Only has an effect when do_basic_tokenize=True.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to tokenize Chinese characters.
            This should likely be deactivated for Japanese,
            see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328.
        do_chinese_wwm (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to do whole word masking for Chinese.
            Chinese sentence will be segmented by a third-party tool first.
            Each substr will be added '##' prefix and its index will be calucated by
            id(##A) = id(A) + vocab_size.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        do_chinese_wwm=False,
        **kwargs,
    ):
        super(BertTokenizer, self).__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a Google pretrained model use "
                "`tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    vocab_file
                )
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            if do_chinese_wwm:
                self.basic_tokenizer = BasicTokenizerWithChineseWWM(
                    do_lower_case=do_lower_case,
                    never_split=never_split,
                    tokenize_chinese_chars=tokenize_chinese_chars,
                )
            else:
                self.basic_tokenizer = BasicTokenizer(
                    do_lower_case=do_lower_case,
                    never_split=never_split,
                    tokenize_chinese_chars=tokenize_chinese_chars,
                )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab.
        For Chinese substr, id = vocab_size + id(substr.remove(##)).
        """
        if _is_chinese_substr(token):
            index = self.vocab.get(token[2:], self.vocab.get(self.unk_token)) + len(self)
        else:
            index = self.vocab.get(token, self.vocab.get(self.unk_token))
        return index

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab.
        For Chinese substr, id = vocab_size + id(substr.remove(##)).
        """
        if index > len(self):
            token = "##" + self.ids_to_tokens.get(index - len(self), self.unk_token)
        else:
            token = self.ids_to_tokens.get(index, self.unk_token)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "")
                + VOCAB_FILES_NAMES["vocab_file"],
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic
    tokenization (punctuation splitting, lower casing, etc.).
    """

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True):
        """Constructs a BasicTokenizer.
        Args:
            **do_lower_case**: Whether to lower case the input.
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level
                (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text.
        Split on "white spaces" only, for sub-word tokenization, see WordPieceTokenizer.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level
                (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class BasicTokenizerWithChineseWWM(BasicTokenizer):
    """Pre-segmentation for Chinese sentences, which will be used in whole word mask."""

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True):
        super(BasicTokenizerWithChineseWWM, self).__init__(
            do_lower_case=do_lower_case,
            never_split=never_split,
            tokenize_chinese_chars=tokenize_chinese_chars,
        )
        try:
            import jieba

            self.pre_tokenizer = lambda x: jieba.lcut(x, HMM=False)
        except ImportError:
            raise (ImportError("Chinese whole word mask need jieba"))

    def _tokenize_chinese_chars(self, text):
        """For Chinese pieces, uses jieba to segment the words and
        adds whitespace around CJK character."""
        output = []
        piece = ""
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                piece += char
            else:
                chinese_words = self.pre_tokenizer(piece)
                for word in chinese_words:
                    output.append(" ")
                    output.append(word)
                    output.append(" ")
                output.append(char)
                piece = ""

        chinese_words = self.pre_tokenizer(piece)
        for word in chinese_words:
            output.append(" ")
            output.append(word)
            output.append(" ")

        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

          input = "有没有"
          output = ["有", "##没", "##有"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.
        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr

                    if substr.startswith("##"):
                        if _is_chinese_substr(substr):
                            if substr[2:] in self.vocab:  # for Chinese substr
                                cur_substr = substr
                                break
                        else:
                            if substr in self.vocab:  # for English substr
                                cur_substr = substr
                                break
                    else:
                        if (
                            substr in self.vocab
                        ):  # non-substr, maybe character or whole Chinese word
                            cur_substr = substr
                            break
                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break

                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
