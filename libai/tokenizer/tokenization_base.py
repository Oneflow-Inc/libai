# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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

"""copy from HuggingFace transformer repo, to tokenize the sentence.
This class only focus on tokenization, converting token to id and their inverse operation.
It does not construct inputs using special symbols."""

import copy
import json
import logging
import os
import unicodedata
from io import open
from typing import Dict, List, Optional, Union

from libai.utils.file_utils import cached_path

logger = logging.getLogger(__name__)


def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


class PreTrainedTokenizer(object):
    """
    Base class for all tokenizers.
    Handle all the shared methods for tokenization and special tokens as well as methods
    dowloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.
    This class also contain the added tokens in a unified way on top of all tokenizers so we don't
    have to handle the specific vocabulary augmentation methods of the various underlying
    dictionary structures (BPE, sentencepiece...).
    Class attributes (overridden by derived classes):
        - ``vocab_files_names``: a python ``dict`` with, as keys, the ``__init__`` keyword name of
        each vocabulary file required by the model, and as associated values, the filename for
        saving the associated file (string).
        - ``pretrained_vocab_files_map``: a python ``dict of dict`` the high-level keys being the
        ``__init__`` keyword name of each vocabulary file required by the model, the low-level
        being the `short-cut-names` (string) of the pretrained models with, as associated values,
        the `url` (string) to the associated pretrained vocabulary file.
        - ``max_model_input_sizes``: a python ``dict`` with, as keys, the `short-cut-names` (string)
        of the pretrained models, and as associated values, the maximum length of the sequence
        inputs of this model, or None if the model has no maximum input size.
        - ``pretrained_init_configuration``: a python ``dict`` with, as keys, the `short-cut-names`
        (string) of the pretrained models, and as associated values, a dictionnary of specific
        arguments to pass to the ``__init__``method of the tokenizer class for this pretrained
        model when loading the tokenizer with the ``from_pretrained()`` method.
    Args:
        bos_token (:obj:`str`, `optional`): A special token representing the beginning of a
        sentence.
        eos_token (:obj:`str`, `optional`): A special token representing the end of a sentence.
        unk_token (:obj:`str`, `optional`): A special token representing an out-of-vocabulary token.
        sep_token (:obj:`str`, `optional`): A special token separating two different sentences in
        the same input (used by BERT for instance).
        pad_token (:obj:`str`, `optional`): A special token used to make arrays of tokens the same
        size for batching purpose. Will then be ignored by attention mechanisms or loss computation.
        cls_token (:obj:`str`, `optional`): A special token representing the class of the input
        (used by BERT for instance).
        mask_token (:obj:`str`, `optional`): A special token representing a masked token (used by
        masked-language modeling pretraining objectives, like BERT).
        eod_token (:obj:`str`, `optional`): A special token representing the end of a document.
        additional_special_tokens (tuple or list of :obj:`str`, `optional`): A tuple or a list of
        additional special tokens.
    """

    vocab_files_names = {}
    pretrained_vocab_files_map = {}
    pretrained_init_configuration = {}
    max_model_input_sizes = {}

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "eod_token",
        "additional_special_tokens",
    ]

    def __init__(self, verbose=True, **kwargs):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._eod_token = None
        self._additional_special_tokens = []
        self.verbose = verbose

        # Added tokens - We store this for both slow and fast tokenizers
        # until the serialization of Fast tokenizers is updated
        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}

        # inputs and kwargs for saving and re-loading
        # (see ``from_pretrained`` and ``save_pretrained``)
        self.init_inputs = ()
        self.init_kwargs = {}

        # We directly set the hidden value to allow initialization with special tokens
        # which are not yet in the vocabulary. Necessary for serialization/de-serialization
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)), f"Value {value} is not a list or tuple"
                    assert all(
                        isinstance(t, str) for t in value
                    ), "One of the tokens is not a string"
                    setattr(self, key, value)
                elif isinstance(value, str):
                    setattr(self, key, value)
                else:
                    raise TypeError(f"special token {key} has to be str but got: {type(value)}")

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        r"""
        Instantiate a :class:`~transformers.PreTrainedTokenizer` (or a derived class) from a
        predefined tokenizer.
        Args:
            pretrained_model_name_or_path: either:
              - a string with the `shortcut name` of a predefined tokenizer to load from cache
              or download, e.g.: ``bert-base-uncased``.
              - a path to a `directory` containing vocabulary files required by the tokenizer,
              for instance saved using the :func:`~transformers.PreTrainedTokenizer.save_pretrained`
              method, e.g.: ``./my_model_directory/``.
              - (not applicable to all derived classes) a path or url to a single saved vocabulary
              file if and only if the tokenizer only requires a single vocabulary file
              (e.g. Bert, XLNet), e.g.: ``./my_model_directory/vocab.txt``.
            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded predefined tokenizer vocabulary files
                should be cached if the standard cache should not be used.
            force_download: (`optional`) boolean, default False:
                Force to (re-)download the vocabulary files and override the cached versions if
                they exists.
            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint,
                e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.
            inputs: (`optional`) positional arguments: will be passed to the
            Tokenizer ``__init__`` method.
            kwargs: (`optional`) keyword arguments: will be passed to the
            Tokenizer ``__init__`` method. Can be used to set special tokens
            like ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``,
            ``pad_token``, ``cls_token``, ``mask_token``, ``additional_special_tokens``.
            See parameters in the doc string of :class:`~transformers.PreTrainedTokenizer`
            for details.
        Examples::
            # We can't instantiate directly the base class `PreTrainedTokenizer` so let's
            # show our examples on a derived class: BertTokenizer
            # Download vocabulary from S3 and cache.
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # If vocabulary files are in a directory (e.g. tokenizer was
            # saved using `save_pretrained('./test/saved_model/')`)
            tokenizer = BertTokenizer.from_pretrained('./test/saved_model/')
            # If the tokenizer uses a single vocabulary file, you can point directly to this file
            tokenizer = BertTokenizer.from_pretrained('./test/saved_model/my_vocab.txt')
            # You can link tokens to special vocabulary when instantiating
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', unk_token='<unk>')
            # You should be sure '<unk>' is in the vocabulary when doing that.
            # Otherwise use tokenizer.add_special_tokens({'unk_token': '<unk>'}) instead)
            assert tokenizer.unk_token == '<unk>'
        """
        return cls._from_pretrained(*inputs, **kwargs)

    @classmethod
    def _from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)

        s3_models = list(cls.max_model_input_sizes.keys())
        vocab_files = {}
        init_configuration = {}
        if pretrained_model_name_or_path in s3_models:
            # Get the vocabulary from AWS S3 bucket
            for file_id, map_list in cls.pretrained_vocab_files_map.items():
                vocab_files[file_id] = map_list[pretrained_model_name_or_path]
            if (
                cls.pretrained_init_configuration
                and pretrained_model_name_or_path in cls.pretrained_init_configuration
            ):
                init_configuration = cls.pretrained_init_configuration[
                    pretrained_model_name_or_path
                ]
        else:
            # Get the vocabulary from local files
            logger.info(
                "Model name '{}' not found in model shortcut name list ({}). "
                "Assuming '{}' is a path or url to a directory containing tokenizer files.".format(
                    pretrained_model_name_or_path,
                    ", ".join(s3_models),
                    pretrained_model_name_or_path,
                )
            )

            # Look for the tokenizer main vocabulary files
            for file_id, file_name in cls.vocab_files_names.items():
                if os.path.isdir(pretrained_model_name_or_path):
                    # If a directory is provided we look for the standard filenames
                    full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
                else:
                    # If a path to a file is provided we use it (will only work for non-BPE
                    # tokenizer using a single vocabulary file)
                    full_file_name = pretrained_model_name_or_path
                if not os.path.exists(full_file_name):
                    logger.info("Didn't find file {}. We won't load it.".format(full_file_name))
                    full_file_name = None
                vocab_files[file_id] = full_file_name

            # Look for the additional tokens files
            additional_files_names = {
                "added_tokens_file": ADDED_TOKENS_FILE,
                "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,
                "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
            }

            # If a path to a file was provided, get the parent directory
            saved_directory = pretrained_model_name_or_path
            if os.path.exists(saved_directory) and not os.path.isdir(saved_directory):
                saved_directory = os.path.dirname(saved_directory)

            for file_id, file_name in additional_files_names.items():
                full_file_name = os.path.join(saved_directory, file_name)
                if not os.path.exists(full_file_name):
                    logger.info("Didn't find file {}. We won't load it.".format(full_file_name))
                    full_file_name = None
                vocab_files[file_id] = full_file_name

            if all(full_file_name is None for full_file_name in vocab_files.values()):
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find tokenizer files"
                    "at this path or url.".format(
                        pretrained_model_name_or_path,
                        ", ".join(s3_models),
                        pretrained_model_name_or_path,
                    )
                )
                return None

        # Get files from url, cache, or disk depending on the case
        try:
            resolved_vocab_files = {}
            for file_id, file_path in vocab_files.items():
                if file_path is None:
                    resolved_vocab_files[file_id] = None
                else:
                    resolved_vocab_files[file_id] = cached_path(
                        file_path,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                    )
        except EnvironmentError as e:
            if pretrained_model_name_or_path in s3_models:
                logger.error("Couldn't reach server to download vocabulary.")
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find files {} "
                    "at this path or url.".format(
                        pretrained_model_name_or_path,
                        ", ".join(s3_models),
                        pretrained_model_name_or_path,
                        str(vocab_files.keys()),
                    )
                )
            raise e

        for file_id, file_path in vocab_files.items():
            if file_path == resolved_vocab_files[file_id]:
                logger.info("loading file {}".format(file_path))
            else:
                logger.info(
                    "loading file {} from cache at {}".format(
                        file_path, resolved_vocab_files[file_id]
                    )
                )

        # Prepare tokenizer initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        tokenizer_config_file = resolved_vocab_files.pop("tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            init_kwargs = json.load(open(tokenizer_config_file, encoding="utf-8"))
            saved_init_inputs = init_kwargs.pop("init_inputs", ())
            if not init_inputs:
                init_inputs = saved_init_inputs
        else:
            init_kwargs = init_configuration

        # Update with newly provided kwargs
        init_kwargs.update(kwargs)

        # Set max length if needed
        if pretrained_model_name_or_path in cls.max_model_input_sizes:
            # if we're using a pretrained model, ensure the tokenizer
            # wont index sequences longer than the number of positional embeddings
            max_len = cls.max_model_input_sizes[pretrained_model_name_or_path]
            if max_len is not None and isinstance(max_len, (int, float)):
                init_kwargs["max_len"] = min(init_kwargs.get("max_len", int(1e12)), max_len)

        # Merge resolved_vocab_files arguments in init_kwargs.
        added_tokens_file = resolved_vocab_files.pop("added_tokens_file", None)
        special_tokens_map_file = resolved_vocab_files.pop("special_tokens_map_file", None)
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path
        if special_tokens_map_file is not None:
            special_tokens_map = json.load(open(special_tokens_map_file, encoding="utf-8"))
            for key, value in special_tokens_map.items():
                if key not in init_kwargs:
                    init_kwargs[key] = value

        # Instantiate tokenizer.
        tokenizer = cls(*init_inputs, **init_kwargs)

        # Save inputs and kwargs for saving and re-loading with ``save_pretrained``
        tokenizer.init_inputs = init_inputs
        tokenizer.init_kwargs = init_kwargs

        # Add supplementary tokens.
        if added_tokens_file is not None:
            added_tok_encoder = json.load(open(added_tokens_file, encoding="utf-8"))
            added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
            tokenizer.added_tokens_encoder.update(added_tok_encoder)
            tokenizer.added_tokens_decoder.update(added_tok_decoder)

        return tokenizer

    def save_pretrained(self, save_directory):
        """Save the tokenizer vocabulary files together with:
            - added tokens,
            - special-tokens-to-class-attributes-mapping,
            - tokenizer instantiation positional and keywords inputs
            (e.g. do_lower_case for Bert).
        This won't save modifications other than (added tokens and special token mapping)
        you may have applied to the tokenizer after the instantiation (e.g. modifying
        tokenizer.do_lower_case after creation).
        This method make sure the full tokenizer can then be re-loaded using the
        :func:`~transformers.PreTrainedTokenizer.from_pretrained` class method.
        """
        if not os.path.isdir(save_directory):
            logger.error("Saving directory ({}) should be a directory".format(save_directory))
            return

        special_tokens_map_file = os.path.join(save_directory, SPECIAL_TOKENS_MAP_FILE)
        added_tokens_file = os.path.join(save_directory, ADDED_TOKENS_FILE)
        tokenizer_config_file = os.path.join(save_directory, TOKENIZER_CONFIG_FILE)

        tokenizer_config = copy.deepcopy(self.init_kwargs)
        tokenizer_config["init_inputs"] = copy.deepcopy(self.init_inputs)
        for file_id in self.vocab_files_names.keys():
            tokenizer_config.pop(file_id, None)

        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tokenizer_config, ensure_ascii=False))

        with open(special_tokens_map_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.special_tokens_map, ensure_ascii=False))

        with open(added_tokens_file, "w", encoding="utf-8") as f:
            if self.added_tokens_encoder:
                out_str = json.dumps(self.added_tokens_encoder, ensure_ascii=False)
            else:
                out_str = "{}"
            f.write(out_str)

        vocab_files = self.save_vocabulary(save_directory)

        return vocab_files + (special_tokens_map_file, added_tokens_file)

    def save_vocabulary(self, save_directory):
        """Save the tokenizer vocabulary to a directory. This method does *NOT* save added tokens
        and special token mappings.
        Please use :func:`~transformers.PreTrainedTokenizer.save_pretrained` `()` to save the
        full Tokenizer state if you want to reload it using the
        :func:`~transformers.PreTrainedTokenizer.from_pretrained` class method.
        """
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Size of the base vocabulary (without the added tokens)."""
        raise NotImplementedError

    def __len__(self):
        """Size of the full vocabulary with the added tokens."""
        return self.vocab_size + len(self.added_tokens_encoder)

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.
        :obj:`tokenizer.get_vocab()[token]` is equivalent to
        :obj:`tokenizer.convert_tokens_to_ids(token)`
        when :obj:`token` is in the vocab.
        Returns:
            :obj:`Dict[str, int]`: The vocabulary.
        """
        raise NotImplementedError

    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.
        Returns:
            :obj:`Dict[str, int]`: The added tokens.
        """
        return self.added_tokens_encoder

    def add_tokens(self, new_tokens: Union[str, List[str]], special_tokens: bool = False) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the
        vocabulary, they are added to it with indices starting from length of
        the current vocabulary.
        .. Note::
            When adding new tokens to the vocabulary, you should make sure to also resize
            the token embedding matrix of the model so that its embedding matrix matches
            the tokenizer.
            In order to do that, please use the
            :meth:`~transformers.PreTrainedModel.resize_token_embeddings` method.
        Args:
            new_tokens (:obj:`str`, or a list of `str`):
                Tokens are only added if they are not already in the vocabulary.
            special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Can be used to specify if the token is a special token. This mostly change
                the normalization behavior
                (special tokens like CLS or [MASK] are usually not lower-cased for instance).
        Returns:
            :obj:`int`: Number of tokens added to the vocabulary.
        Examples::
            # Let's see how to increase the vocabulary of Bert model and tokenizer
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
            print('We have added', num_added_toks, 'tokens')
             # Notice: resize_token_embeddings expect to receive the full size of the new
             # vocabulary, i.e., the length of the tokenizer.
            model.resize_token_embeddings(len(tokenizer))
        """
        if not new_tokens:
            return 0

        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]

        tokens_to_add = []
        for token in new_tokens:
            if not isinstance(token, str):
                raise TypeError(f"Token {token} is not a string but a {type(token)}.")
            if not special_tokens and hasattr(self, "do_lower_case") and self.do_lower_case:
                token = token.lower()
            if (
                token != self.unk_token
                and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
                and token not in tokens_to_add
            ):
                tokens_to_add.append(token)
                if self.verbose:
                    logger.info(f"Adding {token} to the vocabulary")

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(tokens_to_add))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        return len(tokens_to_add)

    def add_special_tokens(self, special_tokens_dict: Dict[str, str]) -> int:
        """
        Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to
        class attributes. If special tokens are NOT in the vocabulary, they are added to it
        (indexed starting from the last index of the current vocabulary).
        .. Note::
            When adding new tokens to the vocabulary, you should make sure to also resize the
            token embedding matrix of the model so that its embedding matrix matches the tokenizer.
            In order to do that, please use the
            :meth:`~transformers.PreTrainedModel.resize_token_embeddings` method.
        Using :obj:`add_special_tokens` will ensure your special tokens can be used in several ways:
        - Special tokens are carefully handled by the tokenizer (they are never split).
        - You can easily refer to special tokens using tokenizer class attributes like
        :obj:`tokenizer.cls_token`. This makes it easy to develop model-agnostic training and
        fine-tuning scripts.
        When possible, special tokens are already registered for provided pretrained models
        (for instance :class:`~transformers.BertTokenizer` :obj:`cls_token` is already registered
        to be :obj`'[CLS]'` and XLM's one is also registered to be :obj:`'</s>'`).
        Args:
            special_tokens_dict (dictionary `str` to `str`):
                Keys should be in the list of predefined special attributes: [``bos_token``,
                ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``,
                ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].
                Tokens are only added if they are not already in the vocabulary (tested by
                checking if the tokenizer assign the index of the ``unk_token`` to them).
        Returns:
            :obj:`int`: Number of tokens added to the vocabulary.
        Examples::
            # Let's see how to add a new classification token to GPT-2
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2Model.from_pretrained('gpt2')
            special_tokens_dict = {'cls_token': '<CLS>'}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print('We have added', num_added_toks, 'tokens')
            # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary,
            # i.e., the length of the tokenizer.
            model.resize_token_embeddings(len(tokenizer))
            assert tokenizer.cls_token == '<CLS>'
        """
        if not special_tokens_dict:
            return 0

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f"Key {key} is not a special token"

            if self.verbose:
                logger.info(f"Assigning {value} to the {key} key of the tokenizer")
            setattr(self, key, value)

            if key == "additional_special_tokens":
                assert isinstance(value, (list, tuple)) and all(
                    isinstance(t, str) for t in value
                ), f"Tokens {value} for key {key} should all be a string"
                added_tokens += self.add_tokens(value, special_tokens=True)
            else:
                assert isinstance(value, str), f"Token {value} for key {key} should be a string"
                added_tokens += self.add_tokens([value], special_tokens=True)

        return added_tokens

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.
        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.
        Args:
            text (:obj:`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific ``prepare_for_tokenization``
                preprocessing method.
        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.strip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text:
                return []
            if not tok_list:
                return self._tokenize(text, **kwargs)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if (
                        sub_text not in self.added_tokens_encoder
                        and sub_text not in self.all_special_tokens
                    ):
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text
            return sum(
                (
                    self._tokenize(token, **kwargs)
                    if token not in self.added_tokens_encoder
                    and token not in self.all_special_tokens
                    else [token]
                    for token in tokenized_text
                ),
                [],
            )

        added_tokens = list(self.added_tokens_encoder.keys()) + self.all_special_tokens
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for
        word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces).
        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Converts a token string (or a sequence of tokens) in a single integer id
        (or a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def encode(self, text):
        if isinstance(text, str):
            tokens = self.tokenize(text)
            return self.convert_tokens_to_ids(tokens)
        elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
            tokens = [self.tokenize(t) for t in text]
            return self.convert_tokens_to_ids(tokens)
        elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
            return text
        else:
            raise ValueError(
                "Input is not valid. Should be a string, a list/tuple of strings or "
                "a list/tuple of integers."
            )

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens,
        using the vocabulary and added tokens.
        Args:
            ids (:obj:`int` or :obj:`List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.
        Returns:
            :obj:`str` or :obj:`List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is
        ``" ".join(tokens)`` but we often want to remove sub-word tokenization artifacts
        at the same time.
        Args:
            tokens (:obj:`List[str]`): The token to join in a string.
        Returns:
            :obj:`str`: The joined tokens.
        """
        return " ".join(tokens)

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
        spaces_between_special_tokens: bool = True,
    ):
        """
        Converts a sequence of ids (integer) in a string, using the tokenizer and vocabulary
        with options to remove special tokens and clean up tokenization spaces.
        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.
        Args:
            token_ids: list of tokenized input ids. Can be obtained using the `encode` or
            `encode_plus` methods.
            skip_special_tokens: if set to True, will replace special tokens.
            clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.
        """
        filtered_tokens = self.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    @property
    def bos_token(self) -> str:
        """
        :obj:`str`: Beginning of sentence token. Log an error if used while not having been set.
        """
        if self._bos_token is None and self.verbose:
            logger.error("Using bos_token, but it is not set yet.")
            return None
        return str(self._bos_token)

    @property
    def eos_token(self) -> str:
        """
        :obj:`str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._eos_token is None and self.verbose:
            logger.error("Using eos_token, but it is not set yet.")
            return None
        return str(self._eos_token)

    @property
    def unk_token(self) -> str:
        """
        :obj:`str`: Unknown token. Log an error if used while not having been set.
        """
        if self._unk_token is None and self.verbose:
            logger.error("Using unk_token, but it is not set yet.")
            return None
        return str(self._unk_token)

    @property
    def sep_token(self) -> str:
        """
        :obj:`str`: Separation token, to separate context and query in an input sequence.
        Log an error if used while not having been set.
        """
        if self._sep_token is None and self.verbose:
            logger.error("Using sep_token, but it is not set yet.")
            return None
        return str(self._sep_token)

    @property
    def pad_token(self) -> str:
        """
        :obj:`str`: Padding token. Log an error if used while not having been set.
        """
        if self._pad_token is None and self.verbose:
            logger.error("Using pad_token, but it is not set yet.")
            return None
        return str(self._pad_token)

    @property
    def cls_token(self) -> str:
        """
        :obj:`str`: Classification token, to extract a summary of an input sequence leveraging
        self-attention along the full depth of the model.
        Log an error if used while not having been set.
        """
        if self._cls_token is None and self.verbose:
            logger.error("Using cls_token, but it is not set yet.")
            return None
        return str(self._cls_token)

    @property
    def mask_token(self) -> str:
        """
        :obj:`str`: Mask token, to use when training a model with masked-language modeling.
        Log an error if used while not having been set.
        """
        if self._mask_token is None and self.verbose:
            logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @property
    def eod_token(self) -> str:
        """
        :obj:`str`: End of document token. Log an error if used while not having been set.
        """
        if self._eod_token is None and self.verbose:
            logger.error("Using eod_token, but it is not set yet.")
            return None
        return str(self._eod_token)

    @property
    def additional_special_tokens(self) -> List[str]:
        """
        :obj:`List[str]`: All the additional special tokens you may want to use.
        Log an error if used while not having been set.
        """
        if self._additional_special_tokens is None and self.verbose:
            logger.error("Using additional_special_tokens, but it is not set yet.")
            return None
        return [str(tok) for tok in self._additional_special_tokens]

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @eod_token.setter
    def eod_token(self, value):
        self._eod_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the beginning of sentence token in the vocabulary.
        Returns :obj:`None` if the token has not been set.
        """
        if self._bos_token is None:
            return None
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the end of sentence token in the vocabulary.
        Returns :obj:`None` if the token has not been set.
        """
        if self._eos_token is None:
            return None
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the unknown token in the vocabulary.
        Returns :obj:`None` if the token has not been set.
        """
        if self._unk_token is None:
            return None
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the separation token in the vocabulary,
        to separate context and query in an input sequence.
        Returns :obj:`None` if the token has not been set.
        """
        if self._sep_token is None:
            return None
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the padding token in the vocabulary.
        Returns :obj:`None` if the token has not been set.
        """
        if self._pad_token is None:
            return None
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def cls_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the classification token in the vocabulary,
        to extract a summary of an input sequence leveraging self-attention
        along the full depth of the model.
        Returns :obj:`None` if the token has not been set.
        """
        if self._cls_token is None:
            return None
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the mask token in the vocabulary, used when training a
        model with masked-language modeling. Returns :obj:`None` if the token has not been set.
        """
        if self._mask_token is None:
            return None
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def eod_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the end of document token in the vocabulary.
        Returns :obj:`None` if the token has not been set.
        """
        if self._eod_token is None:
            return None
        return self.convert_tokens_to_ids(self.eod_token)

    @property
    def additional_special_tokens_ids(self) -> List[int]:
        """
        :obj:`List[int]`: Ids of all the additional special tokens in the vocabulary.
        Log an error if used while not having been set.
        """
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
        """
        A dictionary mapping special token class attributes
        (:obj:`cls_token`, :obj:`unk_token`, etc.) to their values
        (:obj:`'<unk>'`, :obj:`'<cls>'`, etc.).
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self) -> List[str]:
        """
        :obj:`List[str]`: All the special tokens
        (:obj:`'<unk>'`, :obj:`'<cls>'`, etc.) mapped to class attributes.
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + (
                list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value]
            )
        all_toks = list(set(all_toks))
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        """
        :obj:`List[int]`: List the ids of the special tokens
        (:obj:`'<unk>'`, :obj:`'<cls>'`, etc.) mapped to class attributes.
        """
        all_toks = self.all_special_tokens
        all_ids = list(self.convert_tokens_to_ids(all_toks))
        return all_ids

    @staticmethod
    def clean_up_tokenization(out_string):
        """Clean up a list of simple English tokenization artifacts like spaces before
        punctuations and abbreviated forms.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" do not", " don't")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string
