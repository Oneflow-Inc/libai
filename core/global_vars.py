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

from core.tokenizer import build_tokenizer

_GLOBAL_ARGS = None
_GLOBAL_TOKENIZER = None


def get_args():
    """Return arguments."""
    assert _GLOBAL_ARGS is not None, 'args is not initialized.'
    return _GLOBAL_ARGS


def get_tokenizer():
    """Return tokenizer."""
    assert _GLOBAL_TOKENIZER is not None, 'tokenizer is not initialized.'
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER

