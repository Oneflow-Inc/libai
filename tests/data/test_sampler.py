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

import unittest

import numpy as np
import oneflow as flow
from oneflow.utils.data import DataLoader, TensorDataset

from libai.data.samplers import CyclicSampler, SingleRoundSampler


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        super(TestDataLoader, self).setUp()
        self.data = flow.randn(100, 2, 3, 5)
        self.labels = flow.randperm(50).repeat(2)
        self.dataset = TensorDataset(self.data, self.labels)
        self.persistent_workers = False

    def _get_data_loader(self, dataset, **kwargs):
        persistent_workers = kwargs.get('persistent_workers', self.persistent_workers)
        if persistent_workers and kwargs.get('num_workers', 0) == 0:
            persistent_workers = False
        kwargs['persistent_workers'] = persistent_workers
        return DataLoader(dataset, **kwargs)
    
    def _test_sampler(self, **kwargs):
        indices = range(2, 12)  # using a regular iterable
        dl = self._get_data_loader(self.dataset, sampler=indices, batch_size=2, **kwargs)
        self.assertEqual(len(dl), 5)
        for i, (input, _target) in enumerate(dl):
            self.assertEqual(len(input), 2)
            self.assertTrue(
                np.allclose(
                    input.numpy(),
                    self.data[i * 2 + 2: i * 2 + 4, :, :, :].numpy(),
                    atol=1e-4,
                    rtol=1e-4,
                )
            )

    def _test_batch_sampler(self, **kwargs):
        # [(0, 1), (2, 3, 4), (5, 6), (7, 8, 9), ...]
        batches = []  # using a regular iterable
        for i in range(0, 20, 5):
            batches.append(tuple(range(i, i + 2)))
            batches.append(tuple(range(i + 2, i + 5)))

        dl = self._get_data_loader(self.dataset, batch_sampler=batches, **kwargs)
        self.assertEqual(len(dl), 8)
        for i, (input, _target) in enumerate(dl):
            if i % 2 == 0:
                offset = i * 5 // 2
                self.assertEqual(len(input), 2)
                self.assertTrue(
                    np.allclose(
                        input.numpy(),
                        self.data[offset:offset + 2, :, :, :].numpy(),
                        atol=1e-4,
                        rtol=1e-4,
                    )
                )
            else:
                offset = i * 5 // 2
                self.assertEqual(len(input), 3)
                self.assertTrue(
                    np.allclose(
                        input.numpy(),
                        self.data[offset:offset + 3, :, :, :].numpy(),
                        atol=1e-4,
                        rtol=1e-4,
                    )
                )

    def _test_cyclic_sampler(self, **kwargs):
        sampler = CyclicSampler(self.dataset, micro_batch_size=4, shuffle=False, consumed_samples=50, data_parallel_size=1, data_parallel_rank=0)
        dl = self._get_data_loader(self.dataset, batch_sampler=sampler, **kwargs)
        offset = 50
        data = self.data.repeat(10, 1, 1, 1)
        for i, (input, _target) in enumerate(dl):
            self.assertEqual(len(input), 4)
            self.assertTrue(
                np.allclose(
                    input.numpy(),
                    data[i * 4 + offset: i * 4 + 4 + offset, :, :, :].numpy(),
                    atol=1e-4,
                    rtol=1e-4,
                )
            )
            if i > 50:
                break
    
    def _test_single_round_sampler(self, **kwargs):
        sampler = SingleRoundSampler(self.dataset, micro_batch_size=4, shuffle=False, data_parallel_size=1, data_parallel_rank=0, drop_last=False)
        dl = self._get_data_loader(self.dataset, batch_sampler=sampler, **kwargs)
        for i, (input, _target) in enumerate(dl):
            self.assertEqual(len(input), 4)
            self.assertTrue(
                np.allclose(
                    input.numpy(),
                    self.data[i * 4: i * 4 + 4, :, :, :].numpy(),
                    atol=1e-4,
                    rtol=1e-4,
                )
            )

    def test_sampler(self):
        self._test_sampler()
        self._test_sampler(num_workers=4)
        self._test_batch_sampler()
        self._test_batch_sampler(num_workers=4)
        self._test_cyclic_sampler()
        self._test_cyclic_sampler(num_workers=4)
        self._test_single_round_sampler()
        self._test_single_round_sampler(num_workers=4)


if __name__ == '__main__':
    unittest.main()