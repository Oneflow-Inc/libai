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

"""GPT style dataset."""

import logging
import os
import time

import numpy as np
import oneflow as flow

from libai.data.structures import DistTensorData, Instance
from libai.utils import distributed as dist

logger = logging.getLogger(__name__)


class GPT2Dataset(flow.utils.data.Dataset):
    def __init__(
        self,
        name,
        tokenizer,
        data_prefix,
        indexed_dataset,
        max_num_samples,
        max_seq_length,
        seed=1234,
    ):

        self.name = name
        self.tokenizer = tokenizer
        self.indexed_dataset = indexed_dataset

        documents = np.arange(start=0, stop=indexed_dataset.sizes.shape[0], step=1, dtype=np.int32)

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            self.name,
            data_prefix,
            documents,
            self.indexed_dataset.sizes,
            max_num_samples,
            max_seq_length,
            seed,
        )

    def __len__(self):
        # -1 is due to data structure used to retrieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(
                self.doc_idx[doc_index_f], offset=offset_f, length=offset_l - offset_f + 1
            )
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(
                self.indexed_dataset.get(self.doc_idx[doc_index_l], length=offset_l + 1)
            )
            sample = np.concatenate(sample_list)

        input_ids = flow.tensor(np.array(sample[:-1], dtype=np.int64))
        lm_labels = flow.tensor(np.array(sample[1:], dtype=np.int64))
        sample = Instance(
            input_ids=DistTensorData(input_ids),
            labels=DistTensorData(lm_labels, placement_idx=-1),
        )
        return sample


def _build_index_mappings(name, data_prefix, documents, sizes, num_samples, seq_length, seed):
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
    _filename += "_{}_indexmap".format(name)
    _filename += "_{}ns".format(num_samples)
    _filename += "_{}sl".format(seq_length)
    _filename += "_{}s".format(seed)
    doc_idx_filename = _filename + "_doc_idx.npy"
    sample_idx_filename = _filename + "_sample_idx.npy"
    shuffle_idx_filename = _filename + "_shuffle_idx.npy"

    # Build the indexed mapping if not exist.
    # NOTE: use `get_local_rank() == 0` to promise samples will be build in each node.
    if flow.env.get_local_rank() == 0:
        if (
            (not os.path.isfile(doc_idx_filename))
            or (not os.path.isfile(sample_idx_filename))
            or (not os.path.isfile(shuffle_idx_filename))
        ):

            logger.info(
                " > WARNING: could not find index map files, building " "the indices on rank 0 ..."
            )

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            if num_epochs == 1:
                separate_last_epoch = False
                logger.info(" > only one epoch required, setting " "separate_last_epoch to False")

            else:
                # Get the number of samples for the last epoch
                num_samples_from_epochs_minus_one = (
                    (num_epochs - 1) * tokens_per_epoch - 1
                ) // seq_length
                last_epoch_num_samples = num_samples - num_samples_from_epochs_minus_one
                assert (
                    last_epoch_num_samples >= 0
                ), "last epoch number of samples should be non-negative."
                num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
                assert last_epoch_num_samples < (
                    num_samples_per_epoch + 1
                ), "last epoch number of samples exceeded max value."
                # If we have less than 80% of the samples for the last epoch,
                # separate out the epoch and treat it differently.
                # Note: the 80% number is just based on common sense and can
                # be adjusted if needed.
                separate_last_epoch = last_epoch_num_samples < int(0.80 * num_samples_per_epoch)
                if separate_last_epoch:
                    string = (
                        " > last epoch number of samples ({}) is smaller "
                        "than 80% of number of samples per epoch ({}), "
                        "setting separate_last_epoch to True"
                    )
                else:
                    string = (
                        " > last epoch number of samples ({}) is larger "
                        "than 80% of number of samples per epoch ({}), "
                        "setting separate_last_epoch to False"
                    )
                logger.info(string.format(last_epoch_num_samples, num_samples_per_epoch))

            # doc-idx.
            logger.info("start to build and save doc-idx mapping ...")
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            logger.info(
                " > elapsed time to build and save doc-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # sample-idx.

            logger.info("start to build and save sample-idx mapping ...")
            start_time = time.time()

            # Use C++ implementation for speed.
            # First compile and then import.
            from libai.data.data_utils import helpers

            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            sample_idx = helpers.build_sample_idx(
                sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch
            )
            # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                               num_epochs, tokens_per_epoch)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            logger.info(
                " > elapsed time to build and save sample-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retrieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_, sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            logger.info(
                " > elapsed time to build and save shuffle-idx mapping"
                " (seconds): {:4f}".format(time.time() - start_time)
            )

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    dist.synchronize()

    # Load mappings.
    start_time = time.time()
    logger.info(" > loading doc-idx mapping from {}".format(doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode="r")
    logger.info(" > loading sample-idx mapping from {}".format(sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode="r")
    logger.info(" > loading shuffle-idx mapping from {}".format(shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")
    logger.info("    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time))
    logger.info("    total number of samples: {}".format(sample_idx.shape[0]))
    logger.info("    total number of epochs: {}".format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence length, calculate how many
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
    """Build an array with length = number-of-epochs * number-of-documents.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs - 1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    logger.info(
        " > building shuffle index with split [0, {}) and [{}, {}) "
        "...".format(num_samples, num_samples, total_size)
    )

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
