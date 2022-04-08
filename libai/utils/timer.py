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

import time
from time import perf_counter
from numpy import mean
from typing import Optional

import oneflow as flow
from libai.utils import distributed as dist
class Timer:
    """
    A timer which computes the time elapsed since the start/reset of the timer.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the timer.
        """
        self._start = perf_counter()
        self._paused: Optional[float] = None
        self._total_paused = 0
        self._count_start = 1

    def pause(self):
        """
        Pause the timer.
        """
        if self._paused is not None:
            raise ValueError("Trying to pause a Timer that is already paused!")
        self._paused = perf_counter()

    def is_paused(self) -> bool:
        """
        Returns:
            bool: whether the timer is currently paused
        """
        return self._paused is not None

    def resume(self):
        """
        Resume the timer.
        """
        if self._paused is None:
            raise ValueError("Trying to resume a Timer that is not paused!")
        self._total_paused += perf_counter() - self._paused
        self._paused = None
        self._count_start += 1

    def seconds(self) -> float:
        """
        Returns:
            (float): the total number of seconds since the start/reset of the
                timer, excluding the time when the timer is paused.
        """
        if self._paused is not None:
            end_time: float = self._paused  # type: ignore
        else:
            end_time = perf_counter()
        return end_time - self._start - self._total_paused

    def avg_seconds(self) -> float:
        """
        Returns:
            (float): the average number of seconds between every start/reset and
            pause.
        """
        return self.seconds() / self._count_start



class SynchronizedWallClockTimer:
    """Group of timers. Borrowed from Nvidia Megatron code"""
    class Timer:
        """Timer."""
        def __init__(self, name):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()
            self.records = []

        def start(self):
            """Start the timer."""
            assert not self.started_, "timer has already been started"
            dist.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self, reset=False, record=False):
            """Stop the timer."""
            assert self.started_, "timer is not started"
            dist.synchronize()
            if reset:
                self.elapsed_ = time.time() - self.start_time
            else:
                self.elapsed_ += time.time() - self.start_time
            self.started_ = False
            if record:
                self.records.append(self.elapsed_)

        def reset(self):
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False
            self.acc_ = 0.0
            self.cnt_ = 0

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self.elapsed_
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

        def mean(self):
            return trim_mean(self.records, 0.1)

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def get_mean(self, names, normalizer=1.0, reset=True):
        """Get the mean of a group of timers."""
        assert normalizer > 0.0
        means = {}
        for name in names:
            if name in self.timers:
                elapsed_time = (self.timers[name].mean() * 1000.0 / normalizer)
                means[name] = elapsed_time
        return means


def trim_mean(data, trim_percent):
    """Compute the trimmed mean of a list of numbers.

    Args:
        data (list): List of numbers.
        trim_percent (float): Percentage of data to trim.

    Returns:
        float: Trimmed mean.
    """
    assert trim_percent >= 0.0 and trim_percent <= 1.0
    n = len(data)
    data.sort()
    k = int(round(n * (trim_percent)))
    return mean(data[k:n - k])