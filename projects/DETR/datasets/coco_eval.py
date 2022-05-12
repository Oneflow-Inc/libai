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
import flowvision

from libai.evaluation.evaluator import DatasetEvaluator
from libai.utils import distributed as dist
from libai.utils.logger import log_every_n_seconds

from configs.models.configs_detr import postprocessors
from utils.misc import all_gather


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

    def __init__(self, coco_detection):
        
        self.coco_gt = copy.deepcopy(get_coco_api_from_dataset(coco_detection))
        self.iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        
        self._predictions = []
        self.img_ids = []
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval(self.coco_gt, iouType=iou_type)
        self.eval_imgs = {k: [] for k in self.iou_types}
        
    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        """
        orig_target_sizes = flow.stack([t["orig_size"] for t in inputs["labels"]], dim=0)
        # * need a better way to call it 
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        
        if 'segm' in postprocessors.keys():
            target_sizes = flow.stack([t["size"] for t in inputs["labels"]], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        
        predictions = {target['image_id'].item(): output for target, output in zip(inputs["labels"], results)}
        
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]
            
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
                        
            p = coco_eval.params
            # add backward compatibility if useSegm is specified in params
            if p.useSegm is not None:
                p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
                print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
            p.imgIds = list(np.unique(p.imgIds))
            if p.useCats:
                p.catIds = list(np.unique(p.catIds))
            p.maxDets = sorted(p.maxDets)
            coco_eval.params = p 
            
            coco_eval._prepare()
            # loop through images, area range, max detection number
            catIds = p.catIds if p.useCats else [-1]

            if p.iouType == 'segm' or p.iouType == 'bbox':
                computeIoU = coco_eval.computeIoU
            elif p.iouType == 'keypoints':
                computeIoU = coco_eval.computeOks
            coco_eval.ious = {
                (imgId, catId): computeIoU(imgId, catId)
                for imgId in p.imgIds
                for catId in catIds}

            evaluateImg = coco_eval.evaluateImg
            maxDet = p.maxDets[-1]
            evalImgs = [
                evaluateImg(imgId, catId, areaRng, maxDet)
                for catId in catIds
                for areaRng in p.areaRng
                for imgId in p.imgIds
            ]
            # this is NOT in the pycocotools code, but could be done outside
            evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

            self.eval_imgs[iou_type].append(evalImgs)
            
    def evaluate(self):
        """
        Evaluate/summarize the performance after processing all input/output pairs.
        """            
        self.synchronize_between_processes()
        self.accumulate()
        self.summarize()   
        
        return self.coco_eval
        
    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()
    
            
# def pad_batch_images(xi, batch_size, tensor_batch, tensor_micro_batch_size, last_batch_lack, data_parallel_size):
#     import pdb
#     pdb.set_trace()    
#     pad_shape = (batch_size, *xi.shape[1:])
#     local_xi = xi.to_global(
#         sbp=flow.sbp.broadcast, placement=flow.env.all_device_placement("cuda")
#     ).to_local()
#     padded_xi = flow.zeros(pad_shape, dtype=xi.dtype, device="cuda")
#     padded_xi[:tensor_batch, ...] = padded_xi[:tensor_batch, ...] + local_xi
#     for i in range(last_batch_lack - 1):
#         start_idx = tensor_micro_batch_size * (data_parallel_size - i - 1) - 1
#         padded_xi[start_idx:-1] = padded_xi[start_idx + 1 :]
#     padded_xi = padded_xi.to_global(
#         sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), placement=xi.placement
#     ).to_global(sbp=xi.sbp)
    
#     return padded_xi

# def pad_batch_labels(xi,last_batch_lack):
    
#     '''
#     The xi is tuple here.
#     '''
#     padded_xi = []
#     for i in range(last_batch_lack-1):
#         padded_xi.append([])
    
#     padded_xi.extend(xi)
    
#     return tuple(padded_xi)
         
            
# def pad_batch(x_dict, batch_size, last_batch_lack, is_last_batch):

#     tensor_batch = x_dict["images"].tensors.tensor.shape[0]
#     assert tensor_batch <= batch_size
#     if tensor_batch == batch_size and not is_last_batch:
#         return x_dict, batch_size

#     valid_sample = tensor_batch - last_batch_lack
#     data_parallel_size = dist.get_data_parallel_size()
#     assert tensor_batch % data_parallel_size == 0
#     tensor_micro_batch_size = tensor_batch // data_parallel_size
#     padded_dict = {}
    
#     padded_xi = pad_batch_images(x_dict["images"].tensors.tensor, 
#                                batch_size, tensor_batch, tensor_micro_batch_size, 
#                                last_batch_lack, data_parallel_size)
#     import pdb
#     pdb.set_trace()
#     padded_dict["images"] = padded_xi

#     # TODO: labels pad here
#     # padded_xi_list = []
#     # for label_dict in x_dict["labels"]:
#     #     current_label_dict = {}
#     #     for key, xi in label_dict.items():
#     #         print(key)
#     #         import pdb
#     #         pdb.set_trace()
#     #         padded_xi = pad_batch_labels(xi.tensor,last_batch_lack)
#     #         current_label_dict[key] = padded_xi
#     #     padded_xi_list.append(current_label_dict)
#     # padded_dict["labels"] = tuple(padded_xi_list)
#     padded_dict["labels"] = x_dict["labels"]
    
#     return padded_dict, valid_sample


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
    logger.warning(
        f"with eval_iter {eval_iter}, "
        f"reset total samples {len(data_loader.dataset)} to {total_samples}"
    )
    logger.warning(f"Start inference on {total_samples} samples")

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
            # local tensor -> global tensor
            data = get_batch(inputs)
            # is_last_batch = idx == len(data_loader) - 1

            tensor_batch = data["images"].tensors.tensor.shape[0]
            valid_sample = tensor_batch - last_batch_lack
            # TODO: Make sure how to impl pad_batch. Graph mode needs this.
            # paded_data, valid_sample = pad_batch(data, batch_size, last_batch_lack, is_last_batch)
            _, outputs = model(data)
            
            # get valid samplen
            # key: images
            valid_data = {}
            valid_data["images"] = dist.ttol(data["images"].tensors.tensor, ranks=[0] 
                                             if data["images"].tensors.tensor.placement.ranks.ndim == 1 
                                             else [[0]])[:valid_sample]
            
            # *NOTE: dtype of detr label: tuple. len(labels)=bsz, labels[0].dtype=dict
            valid_data["labels"] = []
            for label in data["labels"]:
                label_dict = {}
                for key, value in label.items():
                    label_dict[key] = dist.ttol(value.tensor, ranks=[0] if value.tensor.placement.ranks.ndim == 1 else [[0]])
                valid_data["labels"].append(label_dict)
            valid_data["labels"] = tuple(valid_data["labels"][:valid_sample])
                
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
                    logging.WARNING,
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
    logger.warning("Total valid samples: {}".format(consumed_samples))
    logger.warning(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total_samples - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.warning(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total_samples - num_warmup),
            num_devices,
        )
    )
    
    if evaluator is not None:
        evaluator.evaluate()


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
        
        
def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, flow.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, flowvision.datasets.CocoDetection):
        return dataset.coco
    
def convert_to_xywh(boxes):
    # xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin, ymin, xmax, ymax = boxes.split(1, dim=1)
    xmin, ymin, xmax, ymax = xmin.squeeze(1), ymin.squeeze(1), xmax.squeeze(1), ymax.squeeze(1)    
    return flow.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)