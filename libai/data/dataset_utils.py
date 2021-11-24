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


import numpy as np
import time
import oneflow as flow
import os
from libai.utils import print_rank_0
from global_vars import get_args
from .indexed_dataset import make_dataset as make_indexed_dataset

DSET_TYPE_BERT = "standard_bert"
DSET_TYPE_ICT = "ict"
DSET_TYPE_T5 = "t5"
DSET_TYPE_ROFORMER = "roformer"

DSET_TYPES = [DSET_TYPE_BERT, DSET_TYPE_ICT, DSET_TYPE_T5, DSET_TYPE_ROFORMER]


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
        from . import helpers

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
            # Use C++ implementation for speed.
            # First compile and then import.
            from . import helpers
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
    # 根据总训练epoch数，建立一个epoch*documents大小的一维张量，映射到对应的文档
    # 如果最后一个epoch训练的样本数少于正常epoch训练样本的80%，最后一个单独区分
    # 否则，混合在一起，即把所有要训练的文档混合在一起，然后打乱顺序后一起训练
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
    # 根据样本数建立索引，并打乱顺序
    # 如果区分最后一个，把它的索引单独处理，shuffle后拼接
    # 如果不区分最后一个，对所有样本构建索引，shuffle后返回
    print(' > building shuffle index with split [0, {}) and [{}, {}) '
          '...'.format(num_samples, num_samples, total_size), flush=True)
    
    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples,
                                  step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size,
                                 step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))


def build_train_valid_test_datasets(
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    max_seq_length,
    masked_lm_prob,
    short_seq_prob,
    seed,
    skip_warmup,
    binary_head=False,
    max_seq_length_dec=None,
    dataset_type="standard_bert",
):

    if len(data_prefix) == 1:
        return _build_train_valid_test_datasets(
            data_prefix[0],
            data_impl,
            splits_string,
            train_valid_test_num_samples,
            max_seq_length,
            masked_lm_prob,
            short_seq_prob,
            seed,
            skip_warmup,
            binary_head,
            max_seq_length_dec,
            dataset_type=dataset_type,
        )
    # Blending dataset.
    # Parse the values.
    output = get_datasets_weights_and_num_samples(data_prefix,
                                                  train_valid_test_num_samples)
    prefixes, weights, datasets_train_valid_test_num_samples = output

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(prefixes)):
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            prefixes[i], data_impl, splits_string,
            datasets_train_valid_test_num_samples[i],
            max_seq_length, masked_lm_prob, short_seq_prob,
            seed, skip_warmup, binary_head, dataset_type=dataset_type)
        if train_ds:
            train_datasets.append(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)

        # Blend.
    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(train_datasets, weights)
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(valid_datasets, weights)
    blending_test_dataset = None
    if test_datasets:
        blending_test_dataset = BlendableDataset(test_datasets, weights)

    return (blending_train_dataset, blending_valid_dataset,
            blending_test_dataset)


def _build_train_valid_test_datasets(
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    max_seq_length,
    masked_lm_prob,
    short_seq_prob,
    seed,
    skip_warmup,
    binary_head,
    max_seq_length_dec,
    dataset_type="standard_bert",
):

    if dataset_type not in DSET_TYPES:
        raise ValueError("Invalid dataset_type: ", dataset_type)

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup)

    if dataset_type == DSET_TYPE_ICT:
        args = get_args()
        title_dataset = get_indexed_dataset_(
            args.titles_data_path, data_impl, skip_warmup
        )

    # Get start and end indices of train/valid/train into doc-idx
    # Note that doc-idx is desinged to be num-docs + 1 so we can
    # easily iterate over it.
    total_num_of_documents = indexed_dataset.doc_idx.shape[0] - 1
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
    # 返回一个列表，分别为train、valid、test的index

    # Print stats about the splits.
    print_rank_0(" > dataset split:")

    def print_split_stats(name, index):
        print_rank_0("    {}:".format(name))
        print_rank_0(
            "     document indices in [{}, {}) total of {} "
            "documents".format(
                splits[index], splits[index + 1], splits[index + 1] - splits[index]
            )
        )
        start_index = indexed_dataset.doc_idx[splits[index]]
        end_index = indexed_dataset.doc_idx[splits[index + 1]]
        print_rank_0(
            "     sentence indices in [{}, {}) total of {} "
            "sentences".format(start_index, end_index, end_index - start_index)
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    def build_dataset(index, name):
        # from megatron.data.bert_dataset import BertDataset
        # from megatron.data.ict_dataset import ICTDataset
        # from megatron.data.t5_dataset import T5Dataset
        from .roformer_dataset import RoformerDataset

        dataset = None
        if splits[index + 1] > splits[index]:
            # Get the pointer to the original doc-idx so we can set it later.
            doc_idx_ptr = indexed_dataset.get_doc_idx()
            # Slice the doc-idx
            start_index = splits[index]
            # Add +1 so we can index into the dataset to get the upper bound.
            end_index = splits[index + 1] + 1
            # New doc_idx view.
            indexed_dataset.set_doc_idx(doc_idx_ptr[start_index:end_index])
            # Build the dataset accordingly.
            kwargs = dict(
                name=name,
                data_prefix=data_prefix,
                num_epochs=None,
                max_num_samples=train_valid_test_num_samples[index],
                max_seq_length=max_seq_length,
                seed=seed,
            )

            # if dataset_type == DSET_TYPE_ICT:
            #     args = get_args()
            #     dataset = ICTDataset(
            #         block_dataset=indexed_dataset,
            #         title_dataset=title_dataset,
            #         query_in_block_prob=args.query_in_block_prob,
            #         use_one_sent_docs=args.use_one_sent_docs,
            #         binary_head=binary_head,
            #         **kwargs
            #     )
            # elif dataset_type == DSET_TYPE_T5:
            #     dataset = T5Dataset(
            #         indexed_dataset=indexed_dataset,
            #         masked_lm_prob=masked_lm_prob,
            #         max_seq_length_dec=max_seq_length_dec,
            #         short_seq_prob=short_seq_prob,
            #         **kwargs
            #     )
            # elif dataset_type == DSET_TYPE_BERT:
            #     dataset = BertDataset(
            #         indexed_dataset=indexed_dataset,
            #         masked_lm_prob=masked_lm_prob,
            #         short_seq_prob=short_seq_prob,
            #         binary_head=binary_head,
            #         **kwargs
            #     )
            if dataset_type == DSET_TYPE_ROFORMER:
                dataset = RoformerDataset(
                    indexed_dataset=indexed_dataset,
                    masked_lm_prob=masked_lm_prob,
                    short_seq_prob=short_seq_prob,
                    **kwargs
                )
            else:
                raise NotImplementedError("Dataset type not fully implemented.")

            # Set the original pointer so dataset remains the main dataset.
            indexed_dataset.set_doc_idx(doc_idx_ptr)
            # Checks.
            assert indexed_dataset.doc_idx[0] == 0
            assert indexed_dataset.doc_idx.shape[0] == (total_num_of_documents + 1)
        return dataset

    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "valid")
    test_dataset = build_dataset(2, "test")

    return (train_dataset, valid_dataset, test_dataset)


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):

    print_rank_0(" > building dataset index ...")

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)
    assert indexed_dataset.sizes.shape[0] == indexed_dataset.doc_idx[-1]
    print_rank_0(
        " > finished creating indexed dataset in {:4f} "
        "seconds".format(time.time() - start_time)
    )

    print_rank_0(" > indexed dataset stats:")
    print_rank_0(
        "    number of documents: {}".format(indexed_dataset.doc_idx.shape[0] - 1)
    )
    print_rank_0("    number of sentences: {}".format(indexed_dataset.sizes.shape[0]))

    return indexed_dataset


def get_train_valid_test_split_(splits_string, size):
    """ Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def compile_helper():
    """Compile helper function at runtime. Make sure this
    is invoked on a single process."""
    import os
    import subprocess

    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(["make", "-C", path])
    if ret.returncode != 0:
        print("Making C++ dataset helpers module failed, exiting.")
        import sys

        sys.exit(1)


def get_datasets_weights_and_num_samples(data_prefix,
                                         train_valid_test_num_samples):

    # The data prefix should be in the format of:
    #   weight-1, data-prefix-1, weight-2, data-prefix-2, ..
    assert len(data_prefix) % 2 == 0
    num_datasets = len(data_prefix) // 2
    weights = [0]*num_datasets
    prefixes = [0]*num_datasets
    for i in range(num_datasets):
        weights[i] = float(data_prefix[2*i])
        prefixes[i] = (data_prefix[2*i+1]).strip()
    # Normalize weights
    weight_sum = 0.0
    for weight in weights:
        weight_sum += weight
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]

    # Add 0.5% (the 1.005 factor) so in case the bleding dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    datasets_train_valid_test_num_samples = []
    for weight in weights:
        datasets_train_valid_test_num_samples.append(
            [int(math.ceil(val * weight * 1.005))
             for val in train_valid_test_num_samples])


    return prefixes, weights, datasets_train_valid_test_num_samples
