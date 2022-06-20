from omegaconf import OmegaConf
from typing import List

import oneflow as flow

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


def nested_tensor_from_tensor_list(tensor_list: List[Instance]):
    

    if tensor_list[0].get_fields()["images"].tensor.ndim == 3:

        max_size = _max_by_axis([list(tensor.get_fields()["images"].tensor.shape) for tensor in tensor_list])
        if flow.env.get_world_size() > 1:
            max_size = flow.tensor(max_size).unsqueeze(0)
            max_size = max_size.to_global(sbp=flow.sbp.split(0), placement=flow.placement('cuda', ranks=list(range(flow.env.get_world_size()))))
            max_size = flow.max(max_size, dim=0)[0].numpy().tolist()

        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        
        dtype = tensor_list[0].get_fields()["images"].tensor.dtype
        device = tensor_list[0].get_fields()["images"].tensor.device
        
        tensor = flow.zeros(batch_shape, dtype=dtype, device=device)
        mask = flow.ones((b, h, w), dtype=flow.bool, device=device) 
        
        labels = []
        for i, ins in enumerate(tensor_list):
            img = ins.get_fields()["images"].tensor
            tensor[i, : img.shape[0], : img.shape[1], : img.shape[2]] = img
            mask[i, : img.shape[1], :img.shape[2]] = False
            labels.append(ins.get_fields()["labels"])
        
    else:
        raise ValueError('not supported')
    return Instance(
        images = (DistTensorData(tensor, placement_idx=0), DistTensorData(mask, placement_idx=0)), 
        labels = tuple(labels)
        )

    
def collate_fn(batch):
    assert isinstance(batch[0], Instance), "batch[0] must be `instance`"
    batch = nested_tensor_from_tensor_list(batch)
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
