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

import os
import sys
import time

import oneflow as flow

from .config import parse_args
from libai.tokenizer import build_tokenizer


_GLOBAL_ARGS = None
_GLOBAL_TOKENIZER = None


def get_args():
    """Return arguments."""
    global _GLOBAL_ARGS
    if _GLOBAL_ARGS is None:
        _GLOBAL_ARGS = parse_args()
    return _GLOBAL_ARGS


def get_tokenizer():
    """Return tokenizer."""
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        args = get_args()
        _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER

