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

import oneflow as flow

def print_rank_0(*args, **kwargs):
    if flow.env.get_rank() == 0:
        print(*args, **kwargs)

def print_rank_last(*args, **kwargs):
    if flow.env.get_rank() == flow.env.get_world_size() - 1:
        print(*args, **kwargs)

def print_ranks(ranks, *args, **kwargs):
    rank = flow.env.get_rank()
    if ranks is None:
        ranks = range(flow.env.get_world_size())

    if rank in ranks:
        print(*args, **kwargs)
