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


import datetime
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager

from libai.utils.file_io import PathManager
from libai.utils.history_buffer import HistoryBuffer

__all__ = [
    "get_event_storage",
    "JSONWriter",
    "CommonMetricPrinter",
    "EventStorage",
]

_CURRENT_STORAGE_STACK = []


def get_event_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class:`EventStorage` is currently enabled.
    """
    assert len(
        _CURRENT_STORAGE_STACK
    ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]


class EventWriter:
    """
    Base class for writers that obtain events from :class:`EventStorage` and process them.
    """

    def write(self):
        raise NotImplementedError

    def close(self):
        pass


class JSONWriter(EventWriter):
    """
    Write scalars to a json file.
    It saves scalars as one json per line (instead of a big json) for easy parsing.
    Examples parsing such a json file:
    ::
        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 19,
            "total_loss": 1.9228371381759644,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 39,
            "total_loss": 1.282649278640747,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]
        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...
    """

    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): path to the json file. New data will be appended if the file exists.
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
        """
        self._file_handle = PathManager.open(json_file, "a")
        self._window_size = window_size
        self._last_write = -1

    def write(self):
        storage = get_event_storage()
        to_save = defaultdict(dict)

        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            # keep scalars that have not been written
            if iter <= self._last_write:
                continue
            to_save[iter][k] = v
        if len(to_save):
            all_iters = sorted(to_save.keys())
            self._last_write = max(all_iters)

        for itr, scalars_per_iter in to_save.items():
            scalars_per_iter["iteration"] = itr
            self._file_handle.write(json.dumps(scalars_per_iter, sort_keys=True) + "\n")
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def close(self):
        self._file_handle.close()


class CommonMetricPrinter(EventWriter):
    """
    Print **common** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    It also applies smoothing using a window of 20 elements.
    It's meant to print common metrics in common ways.
    To print something in more customized ways, please implement a similar printer by yourself.
    """

    def __init__(self, batch_size, max_iter):
        """
        Args:
            max_iter (int): the maximum number of iterations to train.
                Used to compute ETA.
        """
        self.logger = logging.getLogger(__name__)
        self._batch_size = batch_size
        self._max_iter = max_iter
        self._last_write = None

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        consumed_samples = storage.samples
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None

        eta_string = None
        try:
            iter_time = storage.history("time").global_avg()
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration - 1)
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            iter_time = None
            # estimate eta on our own - more noisy
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                    iteration - self._last_write[0]
                )
                eta_seconds = estimate_iter_time * (self._max_iter - iteration - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())

        try:
            lr = "{:.2e}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        # TODO: add memory usage statistic
        # if torch.cuda.is_available():
        #     max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        # else:
        max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            " {eta}{iter}  {sample}  {losses}  {time}{data_time}  lr: {lr}  {memory}".format(
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=f"iteration: {iteration}/{self._max_iter}",
                sample=f"consumed samples: {consumed_samples}",
                losses="  ".join(
                    [
                        "{}: {:.4g}".format(k, v.median(200))
                        for k, v in storage.histories().items()
                        if "loss" in k
                    ]
                ),
                time="time: {:.4f}({:.2f})  ".format(iter_time, self._batch_size / iter_time)
                if iter_time is not None
                else "",
                data_time="data_time: {:.4f}".format(data_time) if data_time is not None else "",
                lr=lr,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            )
        )


class EventStorage:
    """
    The user-facing class that provides metric storage functionalities.
    In the future we may add support for storing / logging other types of data if needed.
    """

    def __init__(self, start_iter=0):
        """
        Args:
            start_iter (int): the iteration number to start with
        """
        self._history = defaultdict(HistoryBuffer)
        self._smoothing_hints = {}
        self._latest_scalars = {}
        self._iter = start_iter
        self._batch_size = 0
        self._current_prefix = ""
        self._vis_data = []
        self._histograms = []

    def put_image(self, img_name, img_tensor):
        """
        Add an `img_tensor` associated with `img_name`, to be shown on
        tensorboard.
        Args:
            img_name (str): The name of the image to put into tensorboard.
            img_tensor (torch.Tensor or numpy.array): An `uint8` or `float`
                Tensor of shape `[channel, height, width]` where `channel` is
                3. The image format should be RGB. The elements in img_tensor
                can either have values in [0, 1] (float32) or [0, 255] (uint8).
                The `img_tensor` will be visualized in tensorboard.
        """
        self._vis_data.append((img_name, img_tensor, self._iter))

    def put_scalar(self, name, value, smoothing_hint=True):
        """
        Add a scalar `value` to the `HistoryBuffer` associated with `name`.
        Args:
            smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.
                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal.
        """
        name = self._current_prefix + name
        history = self._history[name]
        value = float(value)
        history.update(value, self._iter)
        self._latest_scalars[name] = (value, self._iter)

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            assert (
                existing_hint == smoothing_hint
            ), "Scalar {} was put with a different smoothing_hint!".format(name)
        else:
            self._smoothing_hints[name] = smoothing_hint

    def put_scalars(self, *, smoothing_hint=True, **kwargs):
        """
        Put multiple scalars from keyword arguments.
        Examples:
            storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
        """
        for k, v in kwargs.items():
            self.put_scalar(k, v, smoothing_hint=smoothing_hint)

    def history(self, name):
        """
        Returns:
            HistoryBuffer: the scalar history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[str -> (float, int)]: mapping from the name of each scalar to the most
                recent value and the iteration number its added.
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(self, window_size=20):
        """
        Similar to :meth:`latest`, but the returned values
        are either the un-smoothed original latest value,
        or a median of the given window_size,
        depend on whether the smoothing_hint is True.
        This provides a default behavior that other writers can use.
        """
        result = {}
        for k, (v, itr) in self._latest_scalars.items():
            result[k] = (
                self._history[k].median(window_size) if self._smoothing_hints[k] else v,
                itr,
            )
        return result

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar
                is noisy and needs smoothing.
        """
        return self._smoothing_hints

    def step(self):
        """
        User should either: (1) Call this function to increment storage.iter when needed. Or
        (2) Set `storage.iter` to the correct iteration number before each iteration.
        The storage will then be able to associate the new data with an iteration number.
        """
        self._iter += 1

    @property
    def iter(self):
        """
        Returns:
            int: The current iteration number. When used together with a trainer,
                this is ensured to be the same as trainer.iter.
        """
        return self._iter

    @iter.setter
    def iter(self, val):
        self._iter = int(val)

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, val):
        self._samples = int(val)

    def __enter__(self):
        _CURRENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_STORAGE_STACK[-1] == self
        _CURRENT_STORAGE_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
        """
        old_prefix = self._current_prefix
        self._current_prefix = name.rstrip("/") + "/"
        yield
        self._current_prefix = old_prefix

    def clear_images(self):
        """
        Delete all the stored images for visualization. This should be called
        after images are written to tensorboard.
        """
        self._vis_data = []

    def clear_histograms(self):
        """
        Delete all the stored histograms for visualization.
        This should be called after histograms are written to tensorboard.
        """
        self._histograms = []
