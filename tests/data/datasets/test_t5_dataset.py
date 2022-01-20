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

from libai.data import T5Dataset
from libai.data.data_utils import get_indexed_dataset
from libai.tokenizer import T5Tokenizer

datat_prefix = "t5_samples_lazy_text_sentence"
tokenizer = T5Tokenizer(vocab_file="spiece.model", bos_token="<s/>")
indexed_dataset = get_indexed_dataset(datat_prefix, data_impl="lazy", skip_warmup=False)

dataset = T5Dataset(
    tokenizer,
    data_prefix=datat_prefix,
    indexed_dataset=indexed_dataset,
)

print(len(indexed_dataset))
print(len(dataset))
print(dataset[0])
