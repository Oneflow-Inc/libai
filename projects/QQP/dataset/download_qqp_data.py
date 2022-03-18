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

import sys

sys.path.append(".")
from libai.utils.file_utils import get_data_from_cache  # noqa

# fmt:off
VOCAB_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/bert-base-chinese-vocab.txt" # noqa
QQP_TRAIN_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/QQP/train.tsv" # noqa
QQP_TEST_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/QQP/dev.tsv" # noqa
# fmt:on


VOCAB_MD5 = "3b5b76c4aef48ecf8cb3abaafe960f09"
QQP_TRAIN_MD5 = "f65950abb9499d8e3e33da7d68d61c4e"
QQP_TEST_MD5 = "35ca3d547003266660a77c6031069548"

cache_dir = "projects/QQP/QQP_DATA/"

if __name__ == "__main__":
    print("downloading vocab...")
    get_data_from_cache(VOCAB_URL, cache_dir, md5=VOCAB_MD5)
    print("downloading training data...")
    get_data_from_cache(QQP_TRAIN_URL, cache_dir, md5=QQP_TRAIN_MD5)
    print("downloading testing data...")
    get_data_from_cache(QQP_TEST_URL, cache_dir, md5=QQP_TEST_MD5)
    print("downloading complete")
