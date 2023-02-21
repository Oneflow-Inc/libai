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

import itertools
import unittest

import oneflow.utils.data as flowdata

from libai.data.samplers import CyclicSampler, SingleRoundSampler


class TestCyclicSampler(unittest.TestCase):
    def test_cyclic_sampler_iterable(self):
        sampler = CyclicSampler(
            list(range(100)),
            micro_batch_size=4,
            shuffle=True,
            consumed_samples=0,
            seed=123,
        )
        output_iter = itertools.islice(sampler, 25)  # iteration=100/4=25
        sample_output = list()
        for batch in output_iter:
            sample_output.extend(batch)
        self.assertEqual(set(sample_output), set(range(100)))

        data_sampler = CyclicSampler(
            list(range(100)),
            micro_batch_size=4,
            shuffle=True,
            consumed_samples=0,
            seed=123,
        )

        data_loader = flowdata.DataLoader(
            list(range(100)), batch_sampler=data_sampler, num_workers=0, collate_fn=lambda x: x
        )

        data_loader_iter = itertools.islice(data_loader, 25)
        output = list()
        for data in data_loader_iter:
            output.extend(data)
        self.assertEqual(output, sample_output)

    def test_cyclic_sampler_seed(self):
        sampler = CyclicSampler(
            list(range(100)),
            micro_batch_size=4,
            shuffle=True,
            seed=123,
        )

        data = list(itertools.islice(sampler, 65))

        sampler = CyclicSampler(
            list(range(100)),
            micro_batch_size=4,
            shuffle=True,
            seed=123,
        )

        data2 = list(itertools.islice(sampler, 65))
        self.assertEqual(data, data2)

    def test_cyclic_sampler_resume(self):
        # Single rank
        sampler = CyclicSampler(
            list(range(10)),
            micro_batch_size=4,
            shuffle=True,
            seed=123,
        )

        all_output = list(itertools.islice(sampler, 50))  # iteration 50 times

        sampler = CyclicSampler(
            list(range(10)),
            micro_batch_size=4,
            shuffle=True,
            seed=123,
            consumed_samples=4 * 11,  # consumed 11 iters
        )

        resume_output = list(itertools.islice(sampler, 39))
        self.assertEqual(all_output[11:], resume_output)

    def test_cyclic_sampler_resume_multi_rank(self):
        # Multiple ranks
        sampler_rank0 = CyclicSampler(
            list(range(10)),
            micro_batch_size=4,
            shuffle=True,
            seed=123,
            data_parallel_rank=0,
            data_parallel_size=2,
        )
        sampler_rank1 = CyclicSampler(
            list(range(10)),
            micro_batch_size=4,
            shuffle=True,
            seed=123,
            data_parallel_rank=1,
            data_parallel_size=2,
        )

        all_output_rank0 = list(itertools.islice(sampler_rank0, 50))  # iteration 50 times
        all_output_rank1 = list(itertools.islice(sampler_rank1, 50))  # iteration 50 times

        sampler_rank0 = CyclicSampler(
            list(range(10)),
            micro_batch_size=4,
            shuffle=True,
            seed=123,
            data_parallel_rank=0,
            data_parallel_size=2,
            consumed_samples=4 * 11,  # consumed 11 iters
        )
        sampler_rank1 = CyclicSampler(
            list(range(10)),
            micro_batch_size=4,
            shuffle=True,
            seed=123,
            data_parallel_rank=1,
            data_parallel_size=2,
            consumed_samples=4 * 11,  # consumed 11 iters
        )

        resume_output_rank0 = list(itertools.islice(sampler_rank0, 39))
        resume_output_rank1 = list(itertools.islice(sampler_rank1, 39))

        self.assertEqual(all_output_rank0[11:], resume_output_rank0)
        self.assertEqual(all_output_rank1[11:], resume_output_rank1)


class TestSingleRoundSampler(unittest.TestCase):
    def test_single_sampler_iterable(self):
        sampler = SingleRoundSampler(
            list(range(100)),
            micro_batch_size=4,
            shuffle=False,
        )
        output_iter = itertools.islice(sampler, 30)  # exceed iteration number
        sample_output = list()
        for batch in output_iter:
            sample_output.extend(batch)
        self.assertEqual(sample_output, list(range(100)))

    def test_single_sampler_multi_rank(self):
        sampler_rank0 = SingleRoundSampler(
            list(range(101)),
            micro_batch_size=4,
            shuffle=False,
            data_parallel_rank=0,
            data_parallel_size=2,
        )
        sampler_rank1 = SingleRoundSampler(
            list(range(101)),
            micro_batch_size=4,
            shuffle=False,
            data_parallel_rank=1,
            data_parallel_size=2,
        )

        output_iter_rank0 = itertools.islice(sampler_rank0, 30)
        sample_output_rank0 = list()
        for batch in output_iter_rank0:
            sample_output_rank0.extend(batch)

        output_iter_rank1 = itertools.islice(sampler_rank1, 30)
        sample_output_rank1 = list()
        for batch in output_iter_rank1:
            sample_output_rank1.extend(batch)

        # Padding 0 if it's not enough for a batch, otherwise `to_global`
        # will raise errors for imbalanced data shape in different ranks
        self.assertEqual(sample_output_rank0, list(range(51)))
        self.assertEqual(sample_output_rank1, list(range(51, 101)) + [0])


if __name__ == "__main__":
    unittest.main()
