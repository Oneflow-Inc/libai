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

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
import time

import oneflow as flow
from omegaconf import OmegaConf

from libai.config import LazyCall

try:
    import nltk

    nltk_available = True
except ImportError:
    nltk_available = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from libai import tokenizer
from libai.data.data_utils import indexed_dataset
from libai.tokenizer import build_tokenizer


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):  # split sentence, tokenize
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.cfg)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params, lang_vars=CustomLanguageVars()
                )
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            text = data[key]
            doc_ids = []
            for sentence in Encoder.splitter.tokenize(text):
                sentence_ids = Encoder.tokenizer.encode(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if (
                len(doc_ids) > 0 and self.args.append_eod
            ):  # append eod token when at the enc of document
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(json_line)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument("--input", type=str, required=True, help="Path to input JSON")
    group.add_argument(
        "--json-keys",
        nargs="+",
        default=["text"],
        help="space separate listed of keys to extract from json",
    )
    group.add_argument(
        "--split-sentences", action="store_true", help="Split documents into sentences."
    )
    group.add_argument(
        "--keep-newlines",
        action="store_true",
        help="Keep newlines between sentences when splitting.",
    )

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-name",
        type=str,
        required=True,
        choices=["BertTokenizer", "GPT2Tokenizer", "T5Tokenizer", "RobertaTokenizer"],
        help="What type of tokenizer to use.",
    )
    group.add_argument("--vocab-file", type=str, default=None, help="Path to the vocab file")
    group.add_argument(
        "--merges-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )
    group.add_argument("--do-lower-case", action="store_true", help="Whether to do lower case.")
    group.add_argument("--extra-ids", type=int, default=0, help="Number of extra ids.")
    group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )
    group.add_argument(
        "--do-chinese-wwm", action="store_true", help="Whether to do whole word mask for Chinese."
    )

    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )
    group.add_argument(
        "--dataset-impl", type=str, default="mmap", choices=["lazy", "cached", "mmap"]
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval between progress updates",
    )
    args = parser.parse_args()

    if args.tokenizer_name.startswith("Bert"):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    return args


def parse_args_to_config(args):

    tokenization = OmegaConf.create()

    tokenization.tokenizer = LazyCall(getattr(tokenizer, args.tokenizer_name))(
        vocab_file="bert-base-chinese-vocab.txt",
        do_lower_case=True,
        do_chinese_wwm=True,
    )

    tokenization.tokenizer.vocab_file = args.vocab_file
    tokenization.tokenizer.do_lower_case = args.do_lower_case
    tokenization.tokenizer.extra_id = args.extra_ids
    tokenization.tokenizer.do_chinese_wwm = args.do_chinese_wwm
    tokenization.append_eod = args.append_eod

    return tokenization


def main():
    args = get_args()
    cfg = parse_args_to_config(args)
    startup_start = time.time()

    print("Opening", args.input)
    fin = open(args.input, "r", encoding="utf-8")

    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)

    encoder = Encoder(args, cfg)
    tokenizer = build_tokenizer(cfg)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, fin, 25)

    level = "document"
    if args.split_sentences:
        level = "sentence"

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix, key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix, key, level)
        builders[key] = indexed_dataset.make_builder(
            output_bin_files[key], impl=args.dataset_impl, vocab_size=len(tokenizer)
        )

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        for key, sentences in doc.items():
            if len(sentences) == 0:
                continue
            for sentence in sentences:
                builders[key].add_item(
                    flow.tensor(sentence, dtype=flow.int32)
                )  # write data into .bin file
            builders[key].end_document()
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(
                f"Processed {i} documents",
                f"({i/elapsed} docs/s, {mbs} MB/s).",
                file=sys.stderr,
            )

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])  # write data into .idx file


if __name__ == "__main__":
    main()
