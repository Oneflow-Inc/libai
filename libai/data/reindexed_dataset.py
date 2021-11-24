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

"""dataset for bert."""

import os
import math
import numpy as np
import oneflow as flow

from libai.utils import print_rank_0
from .indexed_dataset import make_dataset as make_indexed_dataset


def get_samples_mapping(
    indexed_dataset,
    data_prefix,
    num_epochs,
    max_num_samples,
    max_seq_length,
    short_seq_prob,
    seed,
    name,
    binary_head,
):
    """Get a list that maps a sample index to a starting sentence index, end sentence index, and length"""

    if not num_epochs:
        if not max_num_samples:
            raise ValueError("Need to specify either max_num_samples " "or num_epochs")
        num_epochs = np.iinfo(np.int32).max - 1
    if not max_num_samples:
        max_num_samples = np.iinfo(np.int64).max - 1

    # Filename of the index mapping
    indexmap_filename = data_prefix
    indexmap_filename += "_{}_indexmap".format(name)
    if num_epochs != (np.iinfo(np.int32).max - 1):
        indexmap_filename += "_{}ep".format(num_epochs)
    if max_num_samples != (np.iinfo(np.int64).max - 1):
        indexmap_filename += "_{}mns".format(max_num_samples)
    indexmap_filename += "_{}msl".format(max_seq_length)
    indexmap_filename += "_{:0.2f}ssp".format(short_seq_prob)
    indexmap_filename += "_{}s".format(seed)
    indexmap_filename += ".npy"

    # Build the indexed mapping if not exist.
    if flow.env.get_rank() == 0 and not os.path.isfile(indexmap_filename):
        print(
            " > WARNING: could not find index map file {}, building "
            "the indices on rank 0 ...".format(indexmap_filename)
        )

        # Make sure the types match the helpers input types.
        assert indexed_dataset.doc_idx.dtype == np.int64
        assert indexed_dataset.sizes.dtype == np.int32

        # Build samples mapping
        verbose = flow.env.get_rank() == 0
        start_time = time.time()
        print_rank_0(" > building samples index mapping for {} ...".format(name))
        # First compile and then import.
        from libai.data_file import helpers

        samples_mapping = helpers.build_mapping(
            indexed_dataset.doc_idx,  # 包含所有文档序号的向量，一个文档可能对应多个行
            indexed_dataset.sizes,  # 包含每个行的长度的向量
            num_epochs,
            max_num_samples,
            max_seq_length,
            short_seq_prob,
            seed,
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


class ReindexedDataset(flow.utils.data.Dataset):
    def __init__(self, name, indexed_dataset, data_prefix, max_num_samples, num_epochs=None,
                 max_seq_length=512, short_seq_prob=0.01, seed=1234, binary_head=True):
        self.max_seq_length = max_seq_length
        self.binary_head = binary_head

        self.indexed_dataset = indexed_dataset

        self.samples_mapping = get_samples_mapping(self.indexed_dataset,
                                                   data_prefix,
                                                   num_epochs,
                                                   max_num_samples,
                                                   self.max_seq_length, # account for added tokens
                                                   short_seq_prob,
                                                   seed,
                                                   name,
                                                   self.binary_head)
    
    def __len__(self):
        return self.samples_mapping.shape[0]
    
    def __getitem__(self, idx):
        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2 ** 32 since numpy requres the seed to be between 0 and 2 ** 32 - 1
        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2 ** 32))

        assert seq_length <= self.max_seq_length

        if self.binary_head:
            num_sentences = len(sample)
            # assume that we have at least two sentences in the sample
            assert num_sentences > 1, 'make sure each sample has at least two sentences.'

            a_end = 1
            if num_sentences >= 3:
                a_end = np_rng.randint(1, num_sentences)
            tokens_a = []
            for j in range(a_end):
                tokens_a.extend(sample[j])
            
            tokens_b = []
            for j in range(a_end, num_sentences):
                tokens_b.extend(sample[j])
            
            return tokens_a, tokens_b
        else:
            tokens_a = []
            for j in range(len(sample)):
                tokens_a.extend(sample[j])
            return tokens_a


def build_index_mappings(name, data_prefix, documents, sizes,
                          num_samples, seq_length, seed):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}sl'.format(seq_length)
    _filename += '_{}s'.format(seed)
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if flow.env.get_rank() == 0:
        if (not os.path.isfile(doc_idx_filename)) or \
           (not os.path.isfile(sample_idx_filename)) or \
           (not os.path.isfile(shuffle_idx_filename)):

            print_rank_0(' > WARNING: could not find index map files, building '
                         'the indices on rank 0 ...')

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            if num_epochs == 1:
                separate_last_epoch = False
                print(' > only one epoch required, setting '
                      'separate_last_epoch to False', flush=True)

            else:
                # Get the number of samples for the last epoch
                num_samples_from_epochs_minus_one = (
                    (num_epochs - 1) * tokens_per_epoch - 1) // seq_length
                last_epoch_num_samples = num_samples - \
                                         num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, \
                    'last epoch number of samples should be non-negative.'
                num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
                assert last_epoch_num_samples < (num_samples_per_epoch + 1), \
                    'last epoch number of samples exceeded max value.'
                # If we have less than 80% of the samples for the last epoch,
                # seperate out the epoch and treat it differently.
                # Note: the 80% number is just based on common sense and can
                # be adjusted if needed.
                separate_last_epoch = (last_epoch_num_samples <
                                       int(0.80 * num_samples_per_epoch))
                if separate_last_epoch:
                    string = ' > last epoch number of samples ({}) is smaller '\
                             'than 80% of number of samples per epoch ({}), '\
                             'setting separate_last_epoch to True'
                else:
                    string = ' > last epoch number of samples ({}) is larger '\
                             'than 80% of number of samples per epoch ({}), '\
                             'setting separate_last_epoch to False'
                print(string.format(last_epoch_num_samples,
                                    num_samples_per_epoch), flush=True)

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng,
                                     separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save doc-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # sample-idx.
            start_time = time.time()

            from libai.data import helpers
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length,
                                                  num_epochs, tokens_per_epoch)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save sample-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_,
                                             sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))

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
    print_rank_0(' > loading doc-idx mapping from {}'.format(
        doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading sample-idx mapping from {}'.format(
        sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        sample_idx.shape[0]))
    print_rank_0('    total number of epochs: {}'.format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs-1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    print(' > building shuffle index with split [0, {}) and [{}, {}) '
          '...'.format(num_samples, num_samples, total_size), flush=True)
    
    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))


class FullReindexedDataset(flow.utils.data.Dataset):
    def __init__(self, name, indexed_dataset, data_prefix, documents, max_num_samples, num_epochs=None, max_seq_length=512, seed=1234):
        self.indexed_dataset = indexed_dataset

        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        self.doc_idx, self.sample_idx, self.shuffle_ids = build_index_mappings(
            name, data_prefix, documents, self.indexed_dataset.sizes,
            num_samples, max_seq_length, seed)

    def __len__(self):
        return self.sample_idx.shape[0] - 1
    
    def __getitem__(self, idx):
        idx = self.shuffle_idx[idx]
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f, length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(self.indexed_dataset.get(self.doc_idx[doc_index_l], length=offset_l + 1))
            sample = np.concatenate(sample_list)

        return sample
    




