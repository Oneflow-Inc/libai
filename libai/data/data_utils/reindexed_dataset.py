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


import logging
import os
import time

import numpy as np
import oneflow as flow

from libai.data.data_utils import helpers
from libai.utils import distributed as dist

logger = logging.getLogger(__name__)


def get_samples_mapping(data_prefix, indexed_dataset, max_seq_length, short_seq_prob, binary_head):
    """Get a list that maps a sample index to a starting sentence index,
    end sentence index, and length"""

    # Filename of the index mapping
    indexmap_filename = data_prefix
    indexmap_filename += "_{}msl".format(max_seq_length)
    indexmap_filename += "_{}ssp".format(short_seq_prob)
    indexmap_filename += "_sample_mapping.npy"

    documents = indexed_dataset.doc_idx
    sizes = indexed_dataset.sizes

    # Build the indexed mapping if not exist.
    if flow.env.get_rank() == 0 and not os.path.isfile(indexmap_filename):
        logger.info(
            "WARNING: could not find index map file {}, building "
            "the indices on rank 0 ...".format(indexmap_filename)
        )

        # Build samples mapping
        verbose = flow.env.get_rank() == 0
        start_time = time.time()
        logger.info("building samples index mapping for {} ...".format(data_prefix))
        samples_mapping = helpers.build_mapping(
            documents, sizes, max_seq_length, short_seq_prob, verbose, 2 if binary_head else 1,
        )
        logger.info("done building samples index maping")
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        logger.info("saved the index mapping in {}".format(indexmap_filename))
        # Make sure all the ranks have built the mapping
        logger.info(
            "elapsed time to build and save samples mapping "
            "(seconds): {:4f}".format(time.time() - start_time)
        )

    dist.synchronize()

    # Load indexed dataset.
    logger.info("loading indexed mapping from {}".format(indexmap_filename))
    start_time = time.time()
    samples_mapping = np.load(indexmap_filename, allow_pickle=True, mmap_mode="r")
    logger.info("loaded indexed file in {:3.3f} seconds".format(time.time() - start_time))
    logger.info("total number of samples: {}".format(samples_mapping.shape[0]))

    return samples_mapping


class SentenceIndexedDataset(flow.utils.data.Dataset):
    """This class is propused for building sample mapping index from `indexed_dataset` to
    actural dataset.
    It will combine as many consecutive sentences as possible in the same document without
    exceeding `max_seq_length`.
    When it does not reach maximum length, the pad will be filled later. All the sentences in it
    are complete.
    `binary_head` controls whether to return one or two sentences, which will be used in Bert.
    """

    def __init__(
        self,
        data_prefix,
        indexed_dataset,
        max_seq_length=512,
        short_seq_prob=0.0,
        binary_head=False,
    ):
        self.max_seq_length = max_seq_length
        self.short_seq_prob = short_seq_prob
        self.binary_head = binary_head
        if isinstance(indexed_dataset, (list, tuple)):
            self.indexed_dataset = indexed_dataset[0]
            self.align_indexed_dataset = indexed_dataset[1] if len(indexed_dataset) > 1 else None
        else:
            self.indexed_dataset = indexed_dataset
            self.align_indexed_dataset = None

        self.samples_mapping = get_samples_mapping(
            data_prefix,
            self.indexed_dataset,
            self.max_seq_length,
            self.short_seq_prob,
            self.binary_head,
        )

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]
        if self.align_indexed_dataset is not None:
            align_sample = [self.align_indexed_dataset[i] for i in range(start_idx, end_idx)]
            sample = (sample, align_sample)
        assert seq_length <= self.max_seq_length
        return sample

    @property
    def supports_prefetch(self):
        return self.indexed_dataset.supports_prefetch

    def prefetch(self, indices):
        new_indices = []
        for idx in indices:
            start_idx, end_idx, _ = self.samples_mapping[idx]
            new_indices.extend([i for i in range(start_idx, end_idx)])
        self.indexed_dataset.prefetch(new_indices)
        if self.align_indexed_dataset is not None:
            self.align_indexed_dataset(new_indices)


def build_index_mappings(data_prefix, indexed_dataset, max_seq_length):
    """Build sample-idx.
    sample-idx: is the start document index and document offset for each training sample.
    """
    # Filename of the index mappings.
    indexmap_filename = data_prefix
    indexmap_filename += "_{}msl".format(max_seq_length)
    indexmap_filename += "_sample_idx.npy"

    documents = indexed_dataset.doc_idx.astype(np.int64)
    sizes = indexed_dataset.sizes.astype(np.int64)
    num_tokens = np.sum(sizes)

    # Build the indexed mapping if not exist.
    if flow.env.get_rank() == 0 and not os.path.isfile(indexmap_filename):
        logger.info("could not find index map files, building the indices on rank 0 ...")

        # sample-idx.
        start_time = time.time()
        sample_idx = helpers.build_sample_idx(documents, sizes, max_seq_length, num_tokens)
        np.save(indexmap_filename, sample_idx, allow_pickle=True)
        logger.info(
            "elasped time to build and save sample-idx mapping "
            "(seconds): {:4f}".format(time.time() - start_time)
        )

    dist.synchronize()

    # Load mappings.
    start_time = time.time()
    logger.info(" > loading sample-idx mapping from {}".format(indexmap_filename))
    sample_idx = np.load(indexmap_filename, allow_pickle=True, mmap_mode="r")
    logger.info("loaded indexed file in {:3.3f} seconds".format(time.time() - start_time))
    logger.info("total number of samples: {}".format(sample_idx.shape[0]))

    return sample_idx


class BlockIndexedDataset(flow.utils.data.Dataset):
    """This class is propused for building sample mapping index from `indexed_dataset`
    to actural dataset.
    It will extract the sentence with the length of `max_seq_length` from the document.
    If it is less than the maximum length, it will be intercepted from the next document.
    Therefore, it always returns sentences with `max_seq_length`, but it may
    contain incomplete sentences.
    This is used for GPT training, and it can reduce padding and improve training efficiency.
    """

    def __init__(self, data_prefix, indexed_dataset, max_seq_length=512):
        self.max_seq_length = max_seq_length
        if isinstance(indexed_dataset, (list, tuple)):
            self.indexed_dataset = indexed_dataset[0]
            self.align_indexed_dataset = indexed_dataset[1] if len(indexed_dataset) > 1 else None
        else:
            self.indexed_dataset = indexed_dataset
            self.align_indexed_dataset = None

        self.sample_idx = build_index_mappings(
            data_prefix, self.indexed_dataset, self.max_seq_length
        )

    def __len__(self):
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(
                doc_index_f, offset=offset_f, length=offset_l - offset_f + 1
            )
            if self.align_indexed_dataset is not None:
                align_sample = self.align_indexed_dataset.get(
                    doc_index_f, offset=offset_f, length=offset_l - offset_f + 1
                )
                sample = (sample, align_sample)
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(doc_index_f, offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(i))
            # And finally add the relevant portion of last document.
            sample_list.append(self.indexed_dataset.get(doc_index_l, length=offset_l + 1))
            sample = np.concatenate(sample_list)
            if self.align_indexed_dataset is not None:
                align_sample_list = [self.align_indexed_dataset.get(doc_index_f, offset=offset_f)]
                for i in range(doc_index_f + 1, doc_index_l):
                    align_sample_list.append(self.align_indexed_dataset.get(i))
                align_sample_list.append(
                    self.align_indexed_dataset.get(doc_index_l, length=offset_l + 1)
                )
                align_sample = np.concatenate(align_sample_list)
                sample = (sample, align_sample)

        return sample

    @property
    def supports_prefetch(self):
        # this dataset must be `cached`, and IndexedCachedDataset are not support prefetch.
        return False
