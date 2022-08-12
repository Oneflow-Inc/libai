import os.path as osp
import numpy as np
import random
import cv2
import pickle

from PIL import Image
import flowvision.transforms as Transformer
import flowvision
import oneflow.utils.data as data
from libai.data.structures import DistTensorData, Instance


class CityScapes(flowvision.datasets.Cityscapes):
    def __init__(self, root, split = "train", mode = "fine", target_type = "instance", transform = None, target_transform = None, transforms = None):
        super().__init__(root, split, mode, target_type, transform, target_transform, transforms)
        
        
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        
        data_sample = Instance(
            images=DistTensorData(image, placement_idx=0),
            labels=DistTensorData(target, placement_idx=-1),
        )
        
        return data_sample
