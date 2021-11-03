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
import numpy as np
import oneflow as flow
from collections import namedtuple, OrderedDict
from core import print_rank_0, print_rank_last, print_ranks

def item(a):
    """Convert tensor/ndarray to scalar. NOTE: a must has only one element."""
    if isinstance(a, flow.Tensor):
        if a.is_consistent:
            a = a.to_local()
        return a.numpy().item()
    elif isinstance(a, np.ndarray):
        return a.item()
    return a


class Meter(object):
    """Base class for Meters."""
    def __init__(self):
        pass
    
    def reset(self):
        pass
    
    def update(self):
        pass
    
    @property
    def smoothed_value(self):
        raise NotImplementedError
    
    def get_format_str(self, pattern):
        raise NotImplementedError


class SimpleMeter(Meter):
    """Simple meter, which stores the current value."""
    def __init__(self, round=None):
        self.round = round
        self.reset()
    
    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = item(val)
    
    @property
    def smoothed_value(self):
        val = self.val
        if self.round is not None:
            val = round(val, self.round)
        return val

    def get_format_str(self, pattern):
        return pattern.format(self.smoothed_value)


class AverageMeter(Meter):
    """Average meter, which computes and stores the average and current value."""
    def __init__(self, round=None):
        self.round = round
        self.reset()
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = item(val)
        if n > 0:
            self.sum = self.sum + item(val) * n
            self.count = self.count + n
    
    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else self.sum
    
    @property
    def smoothed_value(self):
        val = self.avg
        if self.round is not None and val is not None:
            val = round(val, self.round)
        return val
    
    def get_format_str(self, pattern):
        return pattern.format(self.smoothed_value)


class SumMeter(Meter):
    """Sum meter, which computes and stores the sum."""
    def __init__(self, round=None):
        self.round = round
        self.reset()

    def reset(self):
        self.sum = 0

    def update(self, val):
        self.sum = self.sum + item(val)
    
    @property
    def smoothed_value(self):
        val = self.sum
        if self.round is not None:
            val = round(val, self.round)
        return val
    
    def get_format_str(self, pattern):
        return pattern.format(self.smoothed_value)


class TimeMeter(Meter):
    """Time meter, which computes the average occurrence of some events per second."""
    def __init__(self, round=None):
        self.round = round
        self.start = None
        self.end = None
        self.reset()

    def reset(self):
        self.n = 0
        if self.end is None:
            self.start = time.perf_counter()
        else:
            self.start = self.end
        self.end = None

    def update(self, val):
        self.n = self.n + item(val)

    @property
    def elapsed_time(self):
        return time.perf_counter() - self.start
    
    @property
    def avg(self):
        return self.n / self.elapsed_time()

    @property
    def smoothed_value(self):
        val = self.avg
        if self.round is not None and val is not None:
            val = round(val, self.round)
        return val

    def get_format_str(self, pattern):
        return pattern.format(smoothed_value)


class StopWatchMeter(Meter):
    """Stop and watch meter, which computes the sum/avg duration of some events in seconds."""
    def __init__(self, round=None):
        self.round = round
        self.started = False
        self.start_time = None
        self.end_time = None
        self.reset()

    def start(self):
        self.start_time = time.perf_counter()
        self.started = True

    def stop(self, n=1):
        if self.started:
            elapsed_time = time.perf_counter() - self.start_time
            self.sum = self.sum + elapsed_time
            self.count = self.count + n
            self.started = False

    def reset(self):
        self.sum = 0
        self.count = 0
        self.start()

    def update(self, val):
        self.n = self.n + item(val)

    @property
    def elapsed_time(self):
        return time.perf_counter() - self.start_time
    
    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else self.sum

    @property
    def smoothed_value(self):
        val = self.avg
        if self.round is not None and val is not None:
            val = round(val, self.round)
        return val

    def get_format_str(self, pattern):
        return pattern.format(smoothed_value)


Metric = namedtuple('meter', 'format', 'reset_after_print')

class Logger(object):
    """Used for adding, update and print meters."""
    def __init__(self, rank):
        self.rank = rank
        self.metrics = OrderedDict()

    def register_metric(self, name, meter, print_format=None, reset_after_print=False):
        assert name in self.metrics.keys(), f"{name} is already registered."
        if print_format is not None:
            print_format = name + ": {:3f}"
        self.metrics[name] = Metric(meter, print_format, reset_after_print)

    def metric(self, name):
        assert name not in self.metrics.keys(), f"{name} is not registered."
        return self.metrics[name].meter

    def meter(self, name, *args, **kwargs):
        assert name not in self.metrics.keys(), f"{name} is not registered."
        self.metrics[name].meter.update(*args, **kwargs)

    def print_metrics(self, ranks=None):
        fields = []
        for metric in self.metrics.values():
            meter = metric.meter
            fields.append(meter.get_format_str(metric.format))
            if metric.reset_after_print:
                meter.reset()
        print_ranks(ranks, "[rank:{}] {}".format(self.rank, ", ".join(fields)))

