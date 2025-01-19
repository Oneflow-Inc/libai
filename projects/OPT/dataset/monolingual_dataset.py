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

import numpy as np
import oneflow as flow


class MonolingualDataset(flow.utils.data.Dataset):
    """
    A wrapper around flow.utils.data.Dataset for monolingual data.
    """

    def __init__(
        self,
        dataset,
        sizes,
        src_vocab,
        tgt_vocab=None,
        add_eos_for_other_targets=False,
        shuffle=False,
        add_bos_token=False,
        fixed_pad_length=None,
        pad_to_bsz=None,
        src_lang_idx=None,
        tgt_lang_idx=None,
    ):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = src_vocab
        self.tgt_vocab = tgt_vocab or src_vocab
        self.add_eos_for_other_targets = add_eos_for_other_targets
        self.shuffle = shuffle
        self.add_bos_token = add_bos_token
        self.fixed_pad_length = fixed_pad_length
        self.pad_to_bsz = pad_to_bsz
        self.src_lang_idx = src_lang_idx
        self.tgt_lang_idx = tgt_lang_idx

    def __getitem__(self, index):
        # *future_target* is the original sentence
        # *source* is shifted right by 1 (maybe left-padded with eos)
        #
        # Left-to-right language models should condition on *source* and
        # predict *future_target*.
        source, future_target, _ = self.dataset[index]
        target = self._filter_vocab(future_target)

        source, target = self._maybe_add_bos(source, target)
        return {"id": index, "source": source, "target": target}

    def __len__(self):
        return len(self.dataset)

    def _maybe_add_bos(self, source, target):
        if self.add_bos_token:
            # src_lang_idx and tgt_lang_idx are passed in for multilingual LM, with the
            # first token being a lang_id token.
            bos = self.src_lang_idx or self.vocab.bos()
            source = flow.cat([source.new([bos]), source])
            if target is not None:
                tgt_bos = self.tgt_lang_idx or self.tgt_vocab.bos()
                target = flow.cat([target.new([tgt_bos]), target])
        return source, target

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        return self.sizes[indices]

    def _filter_vocab(self, target):
        if len(self.tgt_vocab) != len(self.vocab):

            def _filter(target):
                mask = target.ge(len(self.tgt_vocab))
                if mask.any():
                    target[mask] = self.tgt_vocab.unk()
                return target

            if isinstance(target, list):
                return [_filter(t) for t in target]
            return _filter(target)
        return target

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        self.dataset.prefetch(indices)
