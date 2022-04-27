import os
import contextlib
import copy
import datetime
import logging
import time
import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from contextlib import ExitStack, contextmanager
from typing import Callable, List, Union
from collections import OrderedDict, abc

import oneflow as flow

from libai.evaluation.evaluator import DatasetEvaluator
from libai.utils import distributed as dist
from libai.utils.logger import log_every_n_seconds

from modeling.post_process import PostProcess

def accuracy(output, target, topk=(1,)):
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        (correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size).item()
        for k in topk
    ]
    
    
class CocoEvaluator(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.
    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, topk=(1, 5)):
        self.topk = topk
        self._predictions = []

    def reset(self):
        self._predictions = []

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
        import pdb
        pdb.set_trace()
        
        pred_logits = outputs["prediction_scores"]
        labels = inputs["labels"]

        # measure accuracy
        topk_acc = accuracy(pred_logits, labels, topk=self.topk)
        num_correct_acc_topk = [acc * labels.size(0) / 100 for acc in topk_acc]

        self._predictions.append(
            {"num_correct_topk": num_correct_acc_topk, "num_samples": labels.size(0)}
        )

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
        if not dist.is_main_process():
            return {}
        else:
            predictions = self._predictions

        total_correct_num = OrderedDict()
        for top_k in self.topk:
            total_correct_num["Acc@" + str(top_k)] = 0

        total_samples = 0
        for prediction in predictions:
            for top_k, num_correct_n in zip(self.topk, prediction["num_correct_topk"]):
                total_correct_num["Acc@" + str(top_k)] += int(num_correct_n)

            total_samples += int(prediction["num_samples"])

        self._results = OrderedDict()
        for top_k, topk_correct_num in total_correct_num.items():
            self._results[top_k] = topk_correct_num / total_samples * 100

        return copy.deepcopy(self._results)


def pad_batch(x_dict, batch_size, last_batch_lack, is_last_batch):
    tensor_batch = x_dict["images"].tensors.tensor.shape[0]
    assert tensor_batch <= batch_size

    if tensor_batch == batch_size and not is_last_batch:
        return x_dict, batch_size

    valid_sample = tensor_batch - last_batch_lack
    data_parallel_size = dist.get_data_parallel_size()
    assert tensor_batch % data_parallel_size == 0
    tensor_micro_batch_size = tensor_batch // data_parallel_size
    padded_dict = {}
    for key, xi in x_dict.items():
        pad_shape = (batch_size, *xi.shape[1:])
        local_xi = xi.to_global(
            sbp=flow.sbp.broadcast, placement=flow.env.all_device_placement("cuda")
        ).to_local()
        padded_xi = flow.zeros(pad_shape, dtype=xi.dtype, device="cuda")
        padded_xi[:tensor_batch, ...] = padded_xi[:tensor_batch, ...] + local_xi
        for i in range(last_batch_lack - 1):
            start_idx = tensor_micro_batch_size * (data_parallel_size - i - 1) - 1
            padded_xi[start_idx:-1] = padded_xi[start_idx + 1 :]
        padded_xi = padded_xi.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), placement=xi.placement
        ).to_global(sbp=xi.sbp)
        padded_dict[key] = padded_xi
    return padded_dict, valid_sample


def inference_on_coco_dataset(
    model,
    data_loader,
    batch_size,
    eval_iter,
    get_batch: Callable,
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
        evaluator = CocoEvaluator([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = CocoEvaluator(evaluator)
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

            data = get_batch(inputs)
            is_last_batch = idx == len(data_loader) - 1
            # TODO: refine the last_batch situation in pad_batch
            paded_data, valid_sample = pad_batch(data, batch_size, last_batch_lack, is_last_batch)
            _, outputs = model(paded_data)
            
            # get valid samplen
            # key: images
            valid_data = {}
            valid_data["images"] = dist.ttol(data["images"].tensors.tensor, ranks=[0] 
                                             if data["images"].tensors.tensor.placement.ranks.ndim == 1 
                                             else [[0]])[:valid_sample]
            
            # *NOTE: dtype of detr label: tuple. len(labels)=bsz
            valid_data["labels"] = []
            for label in data["labels"]:
                label_dict = {}
                for key, value in label.items():
                    label_dict[key] = dist.ttol(value.tensor, ranks=[0] if value.tensor.placement.ranks.ndim == 1 else [[0]])
                valid_data["labels"].append(label_dict)
            valid_data["labels"] = tuple(valid_data["labels"][:valid_sample])
                
            import pdb
            pdb.set_trace()            
            valid_outputs = {}
            # TODO: impl aux_outputs
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