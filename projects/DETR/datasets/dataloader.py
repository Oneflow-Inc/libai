from pickle import TRUE
from omegaconf import OmegaConf
from typing import List

import oneflow as flow

from libai.utils.distributed import get_world_size

from .detection import CocoDetection
from .transforms import make_coco_transforms

from libai.config import LazyCall
from libai.data.build import build_image_train_loader, build_image_test_loader
from libai.data.structures import DistTensorData, Instance


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def padding_tensor_from_tensor_list(tensor_list: List[tuple]):
    
    # image
    # for eager/graph ddp
    # max_size_img = [3, 1333, 1333]
    # max_size_target = 100
    
    # for eager ddp
    max_size_img = _max_by_axis([list(tensor[0].shape) for tensor in tensor_list])
    # switch max_size to global to calculate the maximum shape
    max_size_img = flow.tensor(max_size_img).unsqueeze(0).to_global(
        sbp=flow.sbp.split(0), placement=flow.placement("cuda", ranks=list(range(get_world_size()))))
    max_size_img = max_size_img.max(0)[0].numpy().tolist()

    batch_shape_img = [len(tensor_list)] + max_size_img
    b, c, h, w = batch_shape_img
    dtype = tensor_list[0][0].dtype
    tensor = flow.zeros(batch_shape_img, dtype=dtype)
    tensor_mask = flow.ones((b, h, w), dtype=flow.bool)
    
    # targets
    # for eager ddp
    max_size_target = flow.tensor(max([tensor[1]["boxes"].shape[0] for tensor in tensor_list])).unsqueeze(0).to_global(
    sbp=flow.sbp.split(0), placement=flow.placement("cuda", ranks=list(range(get_world_size()))))
    max_size_target = max_size_target.max(0)[0].numpy().tolist()
    
    boxes = flow.zeros((b, max_size_target, 4), dtype=flow.float32)
    labels = flow.zeros((b, max_size_target), dtype=flow.int64)
    area = flow.zeros((b, max_size_target), dtype=flow.float32)
    iscrowd = flow.zeros((b, max_size_target), dtype=flow.int64)
    orig_size = flow.zeros((b, 2), dtype=flow.int64)
    image_id = flow.zeros(b, dtype=flow.int64)
    size = flow.zeros((b, 2), dtype=flow.int64)
    target_mask = flow.zeros((b, max_size_target), dtype=flow.bool)
    target_orig_size = flow.zeros(b, dtype=flow.int64)

    # labels = []
    for i, sample in enumerate(tensor_list):
        img, targets = sample
        tensor[i, : img.shape[0], : img.shape[1], : img.shape[2]] = img
        tensor_mask[i, : img.shape[1], :img.shape[2]] = False
        valid_length = len(targets["boxes"])
        target_mask[i, :valid_length] = True
        target_orig_size[i] = flow.tensor(valid_length)
        for k, _ in targets.items():
            if k == "boxes":
                boxes[i, :valid_length, :] = targets[k]
            elif k == "labels":
                labels[i, :valid_length] = targets[k]
            elif k == "area":
                area[i, :valid_length] = targets[k]
            elif k == "iscrowd":
                iscrowd[i, :valid_length] = targets[k]
            elif k == "orig_size":
                orig_size[i, :] = targets[k]
            elif k == "size":
                size[i,:] = targets[k]
            elif k == "image_id":
                image_id[i] = targets[k]
                
    return Instance(
        images = (DistTensorData(tensor, placement_idx=0), DistTensorData(tensor_mask, placement_idx=0)), 
        labels = Instance(
            labels = DistTensorData(labels, placement_idx=0),
            boxes = DistTensorData(boxes, placement_idx=0),
            area = DistTensorData(area, placement_idx=0),
            iscrowd = DistTensorData(iscrowd, placement_idx=0),
            orig_size = DistTensorData(orig_size, placement_idx=0),
            size = DistTensorData(size, placement_idx=0),
            image_id = DistTensorData(image_id, placement_idx=0),
            target_mask = DistTensorData(target_mask, placement_idx=0),
            target_orig_size = DistTensorData(target_orig_size, placement_idx=0)
        ))

    
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
    num_workers=0,
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
        num_workers=0,
        collate_fn = collate_fn
    )
]
