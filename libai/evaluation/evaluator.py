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
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import Callable, List, Union

import oneflow as flow

from libai.utils import distributed as dist
from libai.utils.logger import log_every_n_seconds

from .utils import pad_batch

# --------------------------------------------------------
# References:
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/evaluator.py
# --------------------------------------------------------


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.
    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.
    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.

        .. code-block:: python

            pred_logits = outputs["prediction_scores"]
            labels = inputs["labels"]
            # do evaluation on pred_logits/labels pair
            ...

        Args:
            inputs (dict): the inputs that's used to call the model.
            outputs (dict): the return dict of `model(**inputs)`
        """

    def evaluate(self):
        """
        Evaluate/summarize the performance after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., Classification)
                * value: a dict of {metric name: score}, e.g.: {"Acc@1": 75.0}
        """


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.
    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if dist.is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
    model,
    data_loader,
    batch_size,
    eval_iter,
    get_batch: Callable,
    input_placement_device: str,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.
            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        batch_size: batch size for inference
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        eval_iter: running steps for evaluation
        get_batch: a Callable function for getting data from dataloader
        input_placement_device: used in get_batch, set it to `cuda` or `cpu`.
            see input_placement_device in `libai.configs.common.train.py` for more details.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = dist.get_world_size()
    logger = logging.getLogger(__name__)

    total_samples = len(data_loader.dataset)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, len(data_loader) - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    consumed_samples = 0
    dps = dist.get_data_parallel_size()
    last_batch_lack = (dps - (total_samples % dps)) % dps

    # reset total samples
    real_eval_iter = min(eval_iter, len(data_loader))
    total_samples = min(real_eval_iter * batch_size, len(data_loader.dataset))
    logger.info(
        f"with eval_iter {eval_iter}, "
        f"reset total samples {len(data_loader.dataset)} to {total_samples}"
    )
    logger.info(f"Start inference on {total_samples} samples")

    with ExitStack() as stack:
        if isinstance(model, (flow.nn.Module, flow.nn.Graph)):
            stack.enter_context(inference_context(model))
        stack.enter_context(flow.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            if idx >= real_eval_iter:
                break
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            # model forward
            data = get_batch(inputs, input_placement_device)
            is_last_batch = idx == len(data_loader) - 1
            paded_data, valid_sample = pad_batch(data, batch_size, last_batch_lack, is_last_batch)
            outputs = model(**paded_data)

            # get valid sample
            valid_data = {
                key: dist.ttol(value, ranks=[0] if value.placement.ranks.ndim == 1 else [[0]])[
                    :valid_sample
                ]
                for key, value in data.items()
            }
            valid_outputs = {}
            for key, value in outputs.items():
                value = dist.ttol(value, ranks=[0] if value.placement.ranks.ndim == 1 else [[0]])
                if value.ndim > 1:
                    valid_outputs[key] = value[:valid_sample]  # Slice if it's batched output
                else:
                    valid_outputs[key] = value

            if flow.cuda.is_available():
                dist.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            if dist.is_main_process():
                evaluator.process(valid_data, valid_outputs)
            dist.synchronize()
            total_eval_time += time.perf_counter() - start_eval_time

            consumed_samples += valid_sample
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_iter * (total_samples // batch_size - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {consumed_samples}/{total_samples}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info("Total valid samples: {}".format(consumed_samples))
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total_samples - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total_samples - num_warmup),
            num_devices,
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: eager or graph mode in oneflow
    """
    training_mode = model.model.training if isinstance(model, flow.nn.Graph) else model.training
    if isinstance(model, flow.nn.Graph):
        model.model.eval()
    else:
        model.eval()
    yield
    if isinstance(model, flow.nn.Graph):
        model.model.train(training_mode)
    else:
        model.train(training_mode)
