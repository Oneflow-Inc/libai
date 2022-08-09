"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import flowvision
import oneflow as flow
from matplotlib.pyplot import box
from numpy import dtype
from pycocotools import mask as coco_mask


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        # TODO: bugs if return_masks
        mask = flow.as_tensor(mask, dtype=flow.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = flow.stack(masks, dim=0)
    else:
        masks = flow.zeros((0, height, width), dtype=flow.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        image_id = flow.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # Guard against no boxes via resizing
        boxes = flow.as_tensor(boxes, dtype=flow.float32).reshape(-1, 4)
        # BUG: inplace version returns error results
        # boxes[:, 2:] += boxes[:, :2]
        boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]
        # BUG: inplace version returns error results
        # boxes[:, 0::2].clamp_(min=0, max=w)
        # boxes[:, 1::2].clamp_(min=0, max=h)

        # BUG: https://github.com/Oneflow-Inc/oneflow/issues/8834
        # boxes[:, 0::2] = flow.clamp(boxes[:, 0::2], min=0, max=w)
        # boxes[:, 1::2] = flow.clamp(boxes[:, 1::2], min=0, max=h)
        boxes[:, 0] = flow.clamp(boxes[:, 0], min=0, max=w)
        boxes[:, 2] = flow.clamp(boxes[:, 2], min=0, max=w)
        boxes[:, 1] = flow.clamp(boxes[:, 1], min=0, max=h)
        boxes[:, 3] = flow.clamp(boxes[:, 3], min=0, max=h)
        classes = [obj["category_id"] for obj in anno]
        classes = flow.tensor(classes, dtype=flow.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = flow.as_tensor(keypoints, dtype=flow.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = flow.tensor([obj["area"] for obj in anno])
        iscrowd = flow.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = flow.as_tensor([int(h), int(w)])
        target["size"] = flow.as_tensor([int(h), int(w)])

        return image, target


class CocoDetection(flowvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(root=img_folder, annFile=ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks=return_masks)

    def __getitem__(self, idx: int):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # img = flow.ones((3, 873, 1201), dtype=flow.float32)
        # target = {
        #     "boxes": flow.tensor([[0.2107, 0.4918, 0.3994, 0.2495]], dtype=flow.float32),
        #     "labels": flow.tensor([72], dtype=flow.int64),
        #     "image_id": flow.tensor([139], dtype=flow.int64),
        #     "iscrowd": flow.tensor([0], dtype=flow.int64),
        #     "orig_size": flow.tensor([426, 640], dtype=flow.int64),
        #     "size": flow.tensor([873, 1201], dtype=flow.int64),
        #     "area": flow.tensor([46674.9805], dtype=flow.float32),
        # }
        return (img, target)
