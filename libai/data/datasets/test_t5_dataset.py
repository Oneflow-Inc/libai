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

import oneflow as flow

from libai.data.structures import DistTensorData, Instance

class T5Dataset(flow.utils.data.Dataset):
    def __init__(self, vocab_size, num_samples, seq_len):
        self.tokens_enc = flow.randint(0, vocab_size, (num_samples, seq_len))
        self.tokens_dec_in = flow.randint(0, vocab_size, (num_samples, seq_len))
        self.enc_mask = flow.ones(num_samples, seq_len, seq_len).bool()
        self.dec_mask = flow.ones(num_samples, seq_len, seq_len).bool()
        self.enc_dec_mask = flow.ones(num_samples, seq_len, seq_len).bool()
        self.labels = flow.randint(0, vocab_size, (num_samples, seq_len))
        self.loss_mask = flow.randint(0, 2, (num_samples, seq_len)).bool()
                
    def __len__(self):
        return self.tokens_enc.shape[0]

    def __getitem__(self, idx):
        # print(self.tokens_enc[idx].size())
        sample = Instance(
            encoder_input_ids=DistTensorData(self.tokens_enc[idx]),
            decoder_input_ids=DistTensorData(self.tokens_dec_in[idx]),
            encoder_attn_mask=DistTensorData(self.enc_mask[idx]),
            decoder_attn_mask=DistTensorData(self.dec_mask[idx]),
            encoder_decoder_attn_mask=DistTensorData(self.enc_dec_mask[idx]),
            lm_labels=DistTensorData(self.labels[idx], placement_idx=-1),
            loss_mask=DistTensorData(self.loss_mask[idx], placement_idx=-1),
        )
        return sample