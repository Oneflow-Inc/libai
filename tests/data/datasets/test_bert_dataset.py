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

from libai.data.data_utils import get_indexed_dataset
from libai.data.datasets import BertDataset
from libai.tokenizer import BertTokenizer

datat_prefix = "cn_samples_lazy_text_sentence"
tokenizer = BertTokenizer(
    vocab_file="bert-base-chinese-vocab.txt", do_lower_case=True, do_chinese_wwm=True
)
indexed_dataset = get_indexed_dataset(datat_prefix, data_impl="lazy", skip_warmup=False)

dataset = BertDataset(
    tokenizer,
    data_prefix=datat_prefix,
    indexed_dataset=indexed_dataset,
    mask_lm_prob=0.3,
    binary_head=False,
)

print(len(indexed_dataset))
print(len(dataset))
print(dataset[0])
