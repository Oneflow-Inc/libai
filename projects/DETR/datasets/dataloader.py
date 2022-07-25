from pickle import TRUE
from omegaconf import OmegaConf
from typing import List
import numpy as np

import oneflow as flow

from libai.utils.distributed import get_world_size

from .detection import CocoDetection
from .transforms import make_coco_transforms

from libai.config import LazyCall
from libai.data.build import build_image_train_loader, build_image_test_loader
from libai.data.structures import DistTensorData, Instance


# def _max_by_axis(the_list):
#     # type: (List[List[int]]) -> List[int]
#     maxes = the_list[0]
#     for sublist in the_list[1:]:
#         for index, item in enumerate(sublist):
#             maxes[index] = max(maxes[index], item)
#     return maxes


def padding_tensor_from_tensor_list(tensor_list: List[tuple]):
    

    # image padding
    max_size_img = [3, 1334, 1334]
    batch_shape_img = [len(tensor_list)] + max_size_img
    b, c, h, w = batch_shape_img
    dtype = tensor_list[0][0].dtype
    tensor = flow.zeros(batch_shape_img, dtype=dtype)
    tensor_mask = flow.ones((b, h, w), dtype=flow.bool)
    
    # target padding
    max_size_target = 100
    boxes = flow.zeros((b, max_size_target, 4), dtype=flow.float32)
    labels = flow.zeros((b, max_size_target), dtype=flow.int64)
    area = flow.zeros((b, max_size_target), dtype=flow.float32)
    iscrowd = flow.zeros((b, max_size_target), dtype=flow.int64)
    orig_size = flow.zeros((b, 2), dtype=flow.int64)
    image_id = flow.zeros(b, dtype=flow.int64)
    size = flow.zeros((b, 2), dtype=flow.int64)
    target_mask = flow.zeros((b, max_size_target), dtype=flow.bool)
    target_orig_size = flow.zeros(b, dtype=flow.int64)
    
    for i, sample in enumerate(tensor_list):
        img, targets = sample
        # image
        tensor[i, : img.shape[0], : img.shape[1], : img.shape[2]] = img
        tensor_mask[i, : img.shape[1], :img.shape[2]] = False
        # target
        valid_length = len(targets["boxes"])
        target_mask[i, :valid_length] = True
        target_orig_size[i] = flow.tensor(valid_length)
        boxes[i, :valid_length, :] = targets["boxes"]
        labels[i, :valid_length] = targets["labels"]
        area[i, :valid_length] = targets["area"]
        iscrowd[i, :valid_length] = targets["iscrowd"]
        orig_size[i, :] = targets["orig_size"]
        size[i,:] = targets["size"]
        image_id[i] = targets["image_id"]

    return Instance(
        images = DistTensorData(tensor),
        mask = DistTensorData(tensor_mask), 
        labels = DistTensorData(labels),
        boxes = DistTensorData(boxes),
        area = DistTensorData(area),
        iscrowd = DistTensorData(iscrowd),
        orig_size = DistTensorData(orig_size),
        size = DistTensorData(size),
        image_id = DistTensorData(image_id),
        target_mask = DistTensorData(target_mask),
        target_orig_size = DistTensorData(target_orig_size)
        )

    
def collate_fn(batch):
    assert isinstance(batch[0], tuple), "batch[0] must be `instance`"
    batch = padding_tensor_from_tensor_list(batch)
    return batch


dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_image_train_loader)(
    dataset=[
        LazyCall(CocoDetection)(
            img_folder="./dataset",
            ann_file="./dataset/annotations",
            return_masks=False,
            transforms=make_coco_transforms("train"),
        ),
    ],
    num_workers=4,
    mixup_func=None,
    collate_fn = collate_fn
)

dataloader.test = [
    LazyCall(build_image_test_loader)(
        dataset=LazyCall(CocoDetection)(
            img_folder="./dataset",
            ann_file="./dataset/annotations",
            return_masks=False,
            transforms=make_coco_transforms("val"),
        ),
        num_workers=4,
        collate_fn = collate_fn
    )
]
