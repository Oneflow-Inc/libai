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

import logging
from libai.config import instantiate
from libai.utils.registry import Registry

logger = logging.getLogger(__name__)

TOKENIZER_REGISTRY = Registry("tokenizer")
TOKENIZER_REGISTRY.__doc__ = """
Registry for tokenizer, i.e. BertTokenizer.
The registered object will be called with `obj(cfg)` 
and expected to return a `PreTrainedTokenizer` object.
"""

def build_tokenizer(cfg):
    """Initialize tokenizer."""
    if "_target_" in cfg.tokenizer:
        tokenizer = instantiate(cfg.tokenizer)
    else:
        tokenizer_name = cfg.tokenizer.tokenizer_name
        tokenizer = TOKENIZER_REGISTRY.get(tokenizer_name)(**cfg.tokenizer.tokenizer_cfg)
    
    if cfg.tokenizer.append_eod and tokenizer.eod_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.eod_token = tokenizer.eos_token
        else:
            tokenizer.eod_token = tokenizer.pad_token

    # Add vocab size.
    cfg.data.padded_vocab_size = _vocab_size_with_padding(len(tokenizer), cfg)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, cfg):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    padded_vocab_size = orig_vocab_size
    multiple = cfg.data.make_vocab_size_divisible_by * cfg.dist.tensor_parallel_size
    while (padded_vocab_size % multiple) != 0:
        padded_vocab_size += 1
    logger.info(' > padded vocab (size: {}) with {} dummy tokens (new size: {})'.format(
            orig_vocab_size, padded_vocab_size - orig_vocab_size, padded_vocab_size))
    return padded_vocab_size
