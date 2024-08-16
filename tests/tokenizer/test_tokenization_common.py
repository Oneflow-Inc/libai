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


import inspect
import os
import pickle
import re
import shutil
import tempfile
from typing import Tuple

from libai.tokenizer import PreTrainedTokenizer
from tests.fixtures.utils import get_fixtures


def get_tests_dir(append_path=None):
    """
    Args:
        append_path: optional path to append to the tests dir path

    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere.
        Optionally `append_path` is joined after the `tests` dir the former is provided.

    """
    # this function caller's __file__
    caller__file__ = inspect.stack()[1][1]
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))
    if append_path:
        return os.path.join(tests_dir, append_path)
    else:
        return tests_dir


class TokenizerTesterMixin:

    tokenizer_class = None

    def setUp(self):
        self.tokenizers_list = []
        get_fixtures("sample_text.txt")
        with open(f"{get_tests_dir()}/../fixtures/sample_text.txt", encoding="utf-8") as f_data:
            self._data = f_data.read().replace("\n\n", "\n").strip()

        self.tmpdirname = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_txt = self.get_clean_sequence(tokenizer)[0]
        return input_txt, input_txt

    def get_clean_sequence(
        self, tokenizer, with_prefix_space=False, max_length=20
    ) -> Tuple[str, list]:
        toks = [
            (i, tokenizer.decode([i], clean_up_tokenization_spaces=False))
            for i in range(len(tokenizer))
        ]
        toks = list(filter(lambda t: re.match(r"^[ a-zA-Z]+$", t[1]), toks))
        toks = list(filter(lambda t: [t[0]] == tokenizer.encode(t[1]), toks))
        if max_length is not None and len(toks) > max_length:
            toks = toks[:max_length]
        toks_ids = [t[0] for t in toks]

        # Ensure consistency
        output_txt = tokenizer.decode(toks_ids, clean_up_tokenization_spaces=False)
        if " " not in output_txt and len(toks_ids) > 1:
            output_txt = (
                tokenizer.decode([toks_ids[0]], clean_up_tokenization_spaces=False)
                + " "
                + tokenizer.decode(toks_ids[1:], clean_up_tokenization_spaces=False)
            )
        if with_prefix_space:
            output_txt = " " + output_txt
        output_ids = tokenizer.encode(output_txt)
        return output_txt, output_ids

    def get_tokenizers(self, **kwargs):
        return [self.get_tokenizer(**kwargs)]

    def get_tokenizer(self, **kwargs) -> PreTrainedTokenizer:
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def test_tokenizers_common_properties(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                attributes_list = [
                    "bos_token",
                    "eos_token",
                    "unk_token",
                    "sep_token",
                    "pad_token",
                    "cls_token",
                    "mask_token",
                ]
                for attr in attributes_list:
                    self.assertTrue(hasattr(tokenizer, attr))
                    self.assertTrue(hasattr(tokenizer, attr + "_id"))

                self.assertTrue(hasattr(tokenizer, "additional_special_tokens"))
                self.assertTrue(hasattr(tokenizer, "additional_special_tokens_ids"))

                attributes_list = [
                    "init_inputs",
                    "init_kwargs",
                    "added_tokens_encoder",
                    "added_tokens_decoder",
                ]
                for attr in attributes_list:
                    self.assertTrue(hasattr(tokenizer, attr))

    def test_save_and_load_tokenizer(self):
        # Now let's start the test
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                before_tokens = tokenizer.encode(sample_text)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text)
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens, after_tokens)
                self.assertDictEqual(before_vocab, after_vocab)

                shutil.rmtree(tmpdirname)

        # Now let's start the test
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                tokenizer.add_tokens(["bim", "bambam"])
                additional_special_tokens = tokenizer.additional_special_tokens
                additional_special_tokens.append("new_additional_special_token")
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": additional_special_tokens}
                )
                before_tokens = tokenizer.encode(sample_text)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text)
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens, after_tokens)
                self.assertDictEqual(before_vocab, after_vocab)
                self.assertIn("bim", after_vocab)
                self.assertIn("bambam", after_vocab)
                self.assertIn(
                    "new_additional_special_token", after_tokenizer.additional_special_tokens
                )

                shutil.rmtree(tmpdirname)

    def test_pickle_tokenizer(self):
        """Google pickle __getstate__ __setstate__ if you are struggling with this."""
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                self.assertIsNotNone(tokenizer)

                text = "Munich and Berlin are nice cities"
                subwords = tokenizer.tokenize(text)

                filename = os.path.join(self.tmpdirname, "tokenizer.bin")
                with open(filename, "wb") as handle:
                    pickle.dump(tokenizer, handle)

                with open(filename, "rb") as handle:
                    tokenizer_new = pickle.load(handle)

                subwords_loaded = tokenizer_new.tokenize(text)

                self.assertListEqual(subwords, subwords_loaded)

    def test_added_tokens_do_lower_case(self):
        tokenizers = self.get_tokenizers(do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if not hasattr(tokenizer, "do_lower_case") or not tokenizer.do_lower_case:
                    continue

                special_token = tokenizer.all_special_tokens[0]

                text = special_token + " aaaaa bbbbbb low cccccccccdddddddd l " + special_token
                text2 = special_token + " AAAAA BBBBBB low CCCCCCCCCDDDDDDDD l " + special_token

                toks0 = tokenizer.tokenize(text)  # toks before adding new_toks

                new_toks = [
                    "aaaaa bbbbbb",
                    "cccccccccdddddddd",
                    "AAAAA BBBBBB",
                    "CCCCCCCCCDDDDDDDD",
                ]
                added = tokenizer.add_tokens(new_toks)
                self.assertEqual(added, 2)

                toks = tokenizer.tokenize(text)
                toks2 = tokenizer.tokenize(text2)

                self.assertEqual(len(toks), len(toks2))
                self.assertListEqual(toks, toks2)
                self.assertNotEqual(len(toks), len(toks0))  # toks0 should be longer

                # Check that none of the special tokens are lowercased
                sequence_with_special_tokens = (
                    "A " + " yEs ".join(tokenizer.all_special_tokens) + " B"
                )
                tokenized_sequence = tokenizer.tokenize(sequence_with_special_tokens)

                for special_token in tokenizer.all_special_tokens:
                    self.assertTrue(special_token in tokenized_sequence)

        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                special_token = tokenizer.all_special_tokens[0]

                text = special_token + " aaaaa bbbbbb low cccccccccdddddddd l " + special_token
                text2 = special_token + " AAAAA BBBBBB low CCCCCCCCCDDDDDDDD l " + special_token

                new_toks = [
                    "aaaaa bbbbbb",
                    "cccccccccdddddddd",
                    "AAAAA BBBBBB",
                    "CCCCCCCCCDDDDDDDD",
                ]

                toks0 = tokenizer.tokenize(text)  # toks before adding new_toks

                added = tokenizer.add_tokens(new_toks)
                self.assertEqual(added, 4)

                toks = tokenizer.tokenize(text)
                toks2 = tokenizer.tokenize(text2)

                self.assertEqual(len(toks), len(toks2))  # Length should still be the same
                self.assertNotEqual(
                    toks[1], toks2[1]
                )  # But at least the first non-special tokens should differ

                self.assertNotEqual(len(toks), len(toks0))  # toks0 should be longer

    def test_add_tokens_tokenizer(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                # We usually have added tokens from the start in tests
                # because our vocab fixtures are smaller than the original vocabs
                # let's not assert this self.assertEqual(vocab_size, all_size)

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                tokens = tokenizer.encode("aaaaa bbbbbb low cccccccccdddddddd l")

                self.assertGreaterEqual(len(tokens), 4)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

                new_toks_2 = {"eos_token": ">>>>|||<||<<|<<", "pad_token": "<<<<<|||>|>>>>|>"}
                added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
                vocab_size_3 = tokenizer.vocab_size
                all_size_3 = len(tokenizer)

                self.assertNotEqual(vocab_size_3, 0)
                self.assertEqual(vocab_size, vocab_size_3)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l"
                )

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokens[-3])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    def test_add_special_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, ids = self.get_clean_sequence(tokenizer)

                special_token = "[SPECIAL_TOKEN]"

                tokenizer.add_special_tokens({"cls_token": special_token})
                encoded_special_token = tokenizer.encode(special_token)
                self.assertEqual(len(encoded_special_token), 1)

                text = tokenizer.decode(
                    ids + encoded_special_token, clean_up_tokenization_spaces=False
                )
                encoded = tokenizer.encode(text)

                input_encoded = tokenizer.encode(input_text)
                special_token_id = tokenizer.encode(special_token)
                self.assertEqual(encoded, input_encoded + special_token_id)

                decoded = tokenizer.decode(encoded, skip_special_tokens=True)
                self.assertTrue(special_token not in decoded)

    def test_internal_consistency(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, output_text = self.get_input_output_texts(tokenizer)

                tokens = tokenizer.tokenize(input_text)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                ids_2 = tokenizer.encode(input_text)
                self.assertListEqual(ids, ids_2)

                tokens_2 = tokenizer.convert_ids_to_tokens(ids)
                self.assertNotEqual(len(tokens_2), 0)
                text_2 = tokenizer.decode(ids)
                self.assertIsInstance(text_2, str)

                self.assertEqual(text_2, output_text)

    def test_encode_decode_with_spaces(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                new_toks = ["[ABC]", "[DEF]"]
                tokenizer.add_tokens(new_toks)
                input = "[ABC] [DEF] [ABC] [DEF]"
                encoded = tokenizer.encode(input)
                decoded = tokenizer.decode(encoded)
                self.assertEqual(decoded, input)

    def test_pretrained_model_lists(self):
        weights_list = list(self.tokenizer_class.max_model_input_sizes.keys())
        weights_lists_2 = []
        for file_id, map_list in self.tokenizer_class.pretrained_vocab_files_map.items():
            weights_lists_2.append(list(map_list.keys()))

        for weights_list_2 in weights_lists_2:
            self.assertListEqual(weights_list, weights_list_2)

    def test_get_vocab(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab = tokenizer.get_vocab()

                self.assertIsInstance(vocab, dict)
                self.assertEqual(len(vocab), len(tokenizer))

                for word, ind in vocab.items():
                    self.assertEqual(tokenizer.convert_tokens_to_ids(word), ind)
                    self.assertEqual(tokenizer.convert_ids_to_tokens(ind), word)

                tokenizer.add_tokens(["asdfasdfasdfasdf"])
                vocab = tokenizer.get_vocab()
                self.assertIsInstance(vocab, dict)
                self.assertEqual(len(vocab), len(tokenizer))
