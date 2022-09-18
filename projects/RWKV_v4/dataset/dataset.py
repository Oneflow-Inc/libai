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


import json
import random
import os
import numpy as np
import oneflow as flow
import oneflow.nn.functional as F
from oneflow.utils.data import Dataset

from libai.data.structures import DistTensorData, Instance
from .binidx import MMapIndexedDataset

############################################################################################

import math
prime_x = 324331313

def FermatPrimalityTest(number):
    if (number > 1):
        for time in range(3):
            randomNumber = random.randint(2, number)-1
            if (pow(randomNumber, number-1, number) != 1):
                return False
        return True
    else:
        return False

def MillerRabinPrimalityTest(number):
    if number == 2:
        return True
    elif number == 1 or number % 2 == 0:
        return False
    oddPartOfNumber = number - 1
    timesTwoDividNumber = 0
    while oddPartOfNumber % 2 == 0:
        oddPartOfNumber = oddPartOfNumber // 2
        timesTwoDividNumber = timesTwoDividNumber + 1

    for time in range(3):
        while True:
            randomNumber = random.randint(2, number)-1
            if randomNumber != 0 and randomNumber != 1:
                break

        randomNumberWithPower = pow(randomNumber, oddPartOfNumber, number)

        if (randomNumberWithPower != 1) and (randomNumberWithPower != number - 1):
            iterationNumber = 1

            while (iterationNumber <= timesTwoDividNumber - 1) and (randomNumberWithPower != number - 1):
                randomNumberWithPower = pow(randomNumberWithPower, 2, number)
                iterationNumber = iterationNumber + 1
            if (randomNumberWithPower != (number - 1)):
                return False

    return True

############################################################################################

class RWKVDataset(Dataset):
    def __init__(self, datacfg, ctx_len, idx_begin):
        self.ctx_len = ctx_len
        if datacfg[0] == 'idx_bin':
            self.data = MMapIndexedDataset(datacfg[1])
            self.vocab_size = int(os.environ['VOCAB_SIZE'])
            print('current vocab size =', self.vocab_size, "(make sure it's correct)")
            self.data_size = len(self.data._bin_buffer) // 2
            print(f'data has {self.data_size} tokens.')
        elif datacfg[0] == 'numpy':
            self.data = np.load(datacfg[1]).astype('int')
            self.vocab_size = int(os.environ['VOCAB_SIZE'])
            print('current vocab size =', self.vocab_size, "(make sure it's correct)")
            self.data_size = len(self.data)
            print(f'data has {self.data_size} tokens.')
        if datacfg[0] == 'txt':
            self.data = open(datacfg[1], "r", encoding="utf-8").read()
            print('building token list...', end=' ')
            unique = sorted(list(set(self.data)))
            self.vocab_size = len(unique)

            xx = 0
            xxObj = {}
            for u in unique:
                xxObj[xx] = u
                xx += 1

            self.data_size = len(self.data)
            print('data has %d tokens, %d unique.' % (self.data_size, self.vocab_size))
            self.stoi = {ch: i for i, ch in enumerate(unique)}
            self.itos = {i: ch for i, ch in enumerate(unique)}

        self.idx = 0
        self.idx_begin = idx_begin
        self.cuda_id = flow.env.get_rank()
        self.cuda_count = flow.env.get_world_size()
        if datacfg[0] == 'idx_bin':
            self.dataset_slot = self.data_size // ctx_len
            print(self.dataset_slot, 'prime', prime_x, round(prime_x/self.dataset_slot, 4), FermatPrimalityTest(prime_x),
                MillerRabinPrimalityTest(prime_x), (prime_x % 3 != 1))
            assert FermatPrimalityTest(prime_x)
            assert MillerRabinPrimalityTest(prime_x)
            assert prime_x % 3 != 1
            assert prime_x/self.dataset_slot > 0.999999 and prime_x/self.dataset_slot <= 1
        else:
            self.dataset_slot = self.data_size

    def __len__(self):
        return self.dataset_slot

    def __getitem__(self, useless):

        if 'MMapIndexedDataset' in str(type(self.data)):
            idx = self.idx
            need_len = self.ctx_len + 1
            ii = self.idx_begin + (idx * self.cuda_count) + self.cuda_id
            # print(ii)

            factor = (math.sqrt(5) - 1) / 2
            factor = int(prime_x * factor)
            r = ((factor * ii * ii * ii) % prime_x) * self.ctx_len # x^3 requires (prime_x % 3 != 1)

            # print(f'[{self.cuda_id} {ii} {round(r / self.data_size, 3)} {factor}]')#, end='')
            self.idx += 1
            
            dix = self.data.get(idx=0, offset=r, length=need_len).astype(int)
            
            if len(dix) != need_len:
                print(len(dix), need_len, r)
        else:
            #
            # we are cheating: pick a random spot in dataset
            #
            i = np.random.randint(0, self.data_size - (self.ctx_len + 1))
            i = 0

            # i = (self.cuda_id + idx * self.cuda_count) % (self.data_size - (self.ctx_len + 1))
            # i = idx - 2
            # if i < 0:
            #     i = 0
            # print(self.cuda_id, i)

            if 'MMapIndexedDataset' in str(type(self.data)):
                dix = self.data.get(idx=0, offset=i, length=self.ctx_len + 1).astype(int)
            elif 'numpy' in str(type(self.data)):
                dix = self.data[i:i+self.ctx_len+1]
            else:
                dix = [self.stoi[s] for s in self.data[i:i+self.ctx_len+1]]
            # print(dix)

        x = flow.tensor(dix[:-1], dtype=flow.long)
        y = flow.tensor(dix[1:], dtype=flow.long)
        return Instance(idx=DistTensorData(x), targets=DistTensorData(y))


class TOKENIZER:
    def __init__(self, WORD_NAME, UNKNOWN_CHAR=None):
        if 'list' in str(type(WORD_NAME)):
            self.charMode = False
            if WORD_NAME[0] == WORD_NAME[1]:
                from transformers import PreTrainedTokenizerFast
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=WORD_NAME[0])
            else:
                from transformers import GPT2TokenizerFast
                self.tokenizer = GPT2TokenizerFast(WORD_NAME[0], WORD_NAME[1])
        else:
            self.charMode = True
            with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
                self.word_table = json.load(result_file)

            self.vocab_size = len(self.word_table)

            self.stoi = {v: int(k) for k, v in self.word_table.items()}
            self.itos = {int(k): v for k, v in self.word_table.items()}

            self.UNKNOWN_CHAR = self.stoi[UNKNOWN_CHAR]

    def refine_context(self, context):

        context = context.strip().split("\n")
        for c in range(len(context)):
            context[c] = context[c].strip().strip("\u3000").strip("\r")
        context = list(filter(lambda c: c != "", context))
        context = "\n" + ("\n".join(context)).strip()
        if context == "":
            context = "\n"

        return context

    def sample_logits(self, out, x, ctx_len, temperature=1.0, top_p_usual=None, top_p_newline=None):
        # out[self.UNKNOWN_CHAR] = -float('Inf')

        lastChar = int(x[-1])

        probs = F.softmax(flow.tensor(out), dim=-1)

        if self.charMode:
            if self.itos[lastChar] == '\n':
                top_p = top_p_newline
            else:
                top_p = top_p_usual
        else:
            top_p = top_p_usual

        sorted_probs, s_index = flow.sort(probs, descending=True)

        # for j in range(30):
        #     pp = sorted_probs[j].item()
        #     if pp < 0.005:
        #         break
        #     ss = self.itos[int(s_index[j])].replace('\n','_')
        #     print(f'{math.floor(pp*100):>3.0f}{ss}', end='')
        # print('')

        cumulative_probs = flow.cumsum(sorted_probs, dim=-1).numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])

        probs[probs < cutoff] = 0
        # print("[" + str(round(cutoff,4)) + ' ' +
        #       str(round(to_float(sum(probs)),3)) + "]", end = "")

        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)

        return flow.multinomial(probs, num_samples=1)[0]


def to_float(x):
    return x.cpu().detach().numpy().flatten()[0].astype(float)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    flow.manual_seed(seed)
    flow.cuda.manual_seed_all(seed)
