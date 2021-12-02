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
import math
import numpy as np
import oneflow as flow

from libai.data import helpers
from libai.utils import print_rank_0
from .indexed_dataset import make_dataset as make_indexed_dataset


def get_samples_mapping(data_prefix, indexed_dataset, max_seq_length, binary_head):
    """Get a list that maps a sample index to a starting sentence index, end sentence index, and length"""

    # Filename of the index mapping
    indexmap_filename = data_prefix
    indexmap_filename += "_{}msl".format(max_seq_length)
    indexmap_filename += "_sample_mapping.npy"

    documents = indexed_dataset.doc_idx
    sizes = indexed_dataset.sizes

    # Build the indexed mapping if not exist.
    if flow.env.get_rank() == 0 and not os.path.isfile(indexmap_filename):
        print(
            " > WARNING: could not find index map file {}, building "
            "the indices on rank 0 ...".format(indexmap_filename)
        )

        # Make sure the types match the helpers input types.
        assert documents.dtype == np.int64
        assert sizes.dtype == np.int32

        # Build samples mapping
        verbose = flow.env.get_rank() == 0
        start_time = time.time()
        print_rank_0(" > building samples index mapping for {} ...".format(name))
        samples_mapping = helpers.build_mapping(
            documents,  # 包含所有文档序号的向量，一个文档可能对应多个行
            sizes,  # 包含每个行的长度的向量
            max_seq_length,
            verbose,
            2 if binary_head else 1,
        )
        print_rank_0(" > done building samples index maping")
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        print_rank_0(" > saved the index mapping in {}".format(indexmap_filename))
        # Make sure all the ranks have built the mapping
        print_rank_0(
            " > elapsed time to build and save samples mapping "
            "(seconds): {:4f}".format(time.time() - start_time)
        )
    # FIXME(lxy): 这里只考虑了数据并行时的同步问题
    flow._oneflow_internal.eager.multi_client.Sync()
    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model parallel case
    # counts = flow.tensor([1], dtype=flow.long, device="cuda")
    # counts = torch.cuda.LongTensor([1])
    # torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    # torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    # assert counts[0].item() == (
    #     torch.distributed.get_world_size() //
    #     torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

    # Load indexed dataset.
    print_rank_0(" > loading indexed mapping from {}".format(indexmap_filename))
    start_time = time.time()
    samples_mapping = np.load(indexmap_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0(
        "    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time)
    )
    print_rank_0("    total number of samples: {}".format(samples_mapping.shape[0]))

    return samples_mapping


class SentenceIndexedDataset(flow.utils.data.Dataset):
    """ This class is propused for building sample mapping index from `indexed_dataset` to actural dataset.
    It will combine as many consecutive sentences as possible in the same document without exceeding `max_seq_length`.
    When it does not reach maximum length, the pad will be filled later. All the sentences in it are complete. 
    `binary_head` controls whether to return one or two sentences, which will be used in Bert.
    """
    def __init__(self, data_prefix, indexed_dataset, max_seq_length=512, binary_head=False):
        self.max_seq_length = max_seq_length
        self.binary_head = binary_head
        self.indexed_dataset = indexed_dataset

        self.samples_mapping = get_samples_mapping(data_prefix, 
                                                   self.indexed_dataset, 
                                                   self.max_seq_length, 
                                                   self.binary_head)

    def __len__(self):
        return self.samples_mapping.shape[0]
    
    def __getitem__(self, idx):
        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]
        assert seq_length <= self.max_seq_length
        return sample


def build_index_mappings(data_prefix, indexed_dataset, max_seq_length):
    """Build sample-idx.
    sample-idx: is the start document index and document offset for each training sample.
    """
    # Filename of the index mappings.
    indexmap_filename = data_prefix
    indexmap_filename += '_{}ns'.format(num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename = _filename + '_sample_idx.npy'

    documents = indexed_dataset.doc_idx
    sizes = indexed_dataset.sizes
    num_tokens = np.sum(sizes)
    
    # Build the indexed mapping if not exist.
    if flow.env.get_rank() == 0 and not os.path.isfile(indexmap_filename)):
        print_rank_0(' > WARNING: could not find index map files, building '
                        'the indices on rank 0 ...')

        # sample-idx.
        start_time = time.time()
        assert doc_idx.dtype == np.int32
        assert sizes.dtype == np.int32
        sample_idx = helpers.build_sample_idx(documents, sizes, max_seq_length, num_tokens)
        np.save(sample_idx_filename, sample_idx, allow_pickle=True)
        print_rank_0(' > elasped time to build and save sample-idx mapping '
                        '(seconds): {:4f}'.format(time.time() - start_time))

    flow._oneflow_internal.eager.multi_client.Sync()
    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    # counts = torch.cuda.LongTensor([1])
    # torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    # torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    # assert counts[0].item() == (
    #     torch.distributed.get_world_size() //
    #     torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

    # Load mappings.
    start_time = time.time()
    print_rank_0(' > loading sample-idx mapping from {}'.format(
        indexmap_filename))
    sample_idx = np.load(indexmap_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(sample_idx.shape[0]))

    return sample_idx


class DocumentIndexedDataset(flow.utils.data.Dataset):
    """ This class is propused for building sample mapping index from `indexed_dataset` to actural dataset.
    It will extract the sentence with the length of `max_seq_length` from the document. 
    If it is less than the maximum length, it will be intercepted from the next document. 
    Therefore, it always returns sentences with `max_seq_length`, but it may contain incomplete sentences.
    This is used for GPT training, and it can reduce padding and improve training efficiency.
    """
    def __init__(self, data_prefix, indexed_dataset, max_seq_length=512):
        self.max_seq_length = max_seq_length
        self.indexed_dataset = indexed_dataset

        self.sample_idx = build_index_mappings(data_prefix, 
                                               self.indexed_dataset,
                                               self.max_seq_length)

    def __len__(self):
        return self.sample_idx.shape[0] - 1
    
    def __getitem__(self, idx):
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(doc_index_f, offset=offset_f, length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(doc_index_f, offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(i))
            # And finally add the relevant portion of last document.
            sample_list.append(self.indexed_dataset.get(doc_index_l, length=offset_l + 1))
            sample = np.concatenate(sample_list)

        return sample
    
