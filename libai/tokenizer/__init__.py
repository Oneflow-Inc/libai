# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from libai.utils import print_rank_0
from .tokenization_bert import BertTokenizer
from .tokenization_gpt2 import GPT2Tokenizer

def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type),
              flush=True)

    # Select and instantiate the tokenizer.
    assert args.vocab_file is not None
    if args.tokenizer_type == 'BertWordPieceLowerCase':
        tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    elif args.tokenizer_type == 'BertWordPieceCase':
        tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=False)
    elif args.tokenizer_type == 'GPT2BPETokenizer':
        assert args.merge_file is not None
        tokenizer = GPT2Tokenizer(args.vocab_file, args.merge_file)
    else:
        raise NotImplementedError

    if args.append_eod and tokenizer.eod_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.eod_token = tokenizer.eos_token
        else:
            tokenizer.eod_token = tokenizer.pad_token

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(len(tokenizer), args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    padded_vocab_size = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    while (padded_vocab_size % multiple) != 0:
        padded_vocab_size += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens (new size: {})'.format(
            orig_vocab_size, padded_vocab_size - orig_vocab_size, padded_vocab_size), flush=True)
    return padded_vocab_size

