# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
import mmap
import pickle as pkl
from itertools import accumulate
from multiprocessing import Lock

import oneflow as flow


def get_lazy_path(path):
    return os.path.splittext(path)[0] + '.lazy'

def split_strings(strings, start, chr_lens):
    return [strings[i - start: j - start] for i, j in zip([start] + chr_lens[:-1], chr_lens)]

def make_lazy(path, strs, data_type='data'):
    lazypath = get_lazy_path(path)
    if not os.path.exists(lazypath):
        os.makedirs(lazypath)
    datapath = os.path.join(lazypath, data_type)
    lenpath = os.path.join(lazypath, data_type + '.len.pkl')
    if flow.env.get_rank() == 0:
        with open(datapath, 'wb') as f:
            str_lens = []
            str_cnt = 0
            for s in strs:
                if isinstance(s, dict):
                    s = s['text']
                encoded = s.encode('utf-8')
                f.write(encoded)
                str_cnt = len(encoded)
                str_lens.append(str_cnt)
        pkl.dump(str_lens, open(lenpath, 'wb'))
    else:
        while not os.path.exists(lenpath):
            time.sleep(1)



class lazy_loader(object):
    def __init__(self, path, data_type='data', mem_map=False, map_fn=None):
        lazypath = get_lazy_path(path)
        datapath = os.path.join(lazypath, data_type)
        lenpath = os.path.join(lazypath, data_type + '.len.pkl')
        
        self._file = open(datapath, 'rb', buffering=0)
        self.file = self._file
        self.mem_map = mem_map
        if self.mem_map:
            self.file = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)

        self.lens = pkl.load(open(lenpath, 'rb'))
        self.ends = list(accumulate(self.lens))
        self.dumb_ends = list(self.ends)
        self.read_lock = Lock()
    
    def __getitem__(self, index):
        if not isinstance(index, slice):
            if index == 0:
                start = 0
            else:
                start = self.ends[index - 1]
            end = self.ends[index]
            rtn = self.file_read(start, end)
        else:
            chr_lens = self.ends[index]
            if index.start == 0 or index.start is None:
                start = 0
            else:
                start = self.ends[index.start - 1]
            stop = chr_lens[-1]
            strings = self.file_read(start, stop)
            rtn = split_strings(strings, start, chr_lens)
        return rtn
    
    def __len__(self):
        return len(self.ends)
    
    def file_read(self, start=0, end=None):
        self.read_lock.acquire()
        self.file.seek(start)
        if end is None:
            rtn = self.file.read()
        else:
            rtn = self.file.read(end - start)
        self.read_lock.release()
        rtn = rtn.decode('utf-8', 'ignore')
        if self.mem_map:
            rtn = rtn.decode('unicode_escape')
        return rtn

