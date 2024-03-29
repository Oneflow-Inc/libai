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

logger = logging.getLogger(__name__)


def build_tokenizer(cfg):
    """Initialize tokenizer."""
    tokenizer = instantiate(cfg.tokenizer)

    if cfg.get("append_eod", None) and tokenizer.eod_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.eod_token = tokenizer.eos_token
        else:
            tokenizer.eod_token = tokenizer.pad_token

    return tokenizer
