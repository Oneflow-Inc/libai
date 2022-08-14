import os.path as osp
import numpy as np
import random
import cv2
import pickle

from PIL import Image
import oneflow as flow
import flowvision.transforms as Transformer
import flowvision
import oneflow.utils.data as data
from libai.data.structures import DistTensorData, Instance


class CityScapes(flowvision.datasets.Cityscapes):
    
    color2index = {}
    color2index[(0,0,0)] = 0
    for obj in flowvision.datasets.Cityscapes.classes:
        if obj.ignore_in_eval:
            continue
        idx = obj.train_id
        label = obj.name
        color = obj.color
        color2index[color] = idx
        
    def __init__(self, root, split = "train", mode = "fine", target_type = "instance", transform = None, target_transform = None, transforms = None):
        super().__init__(root, split, mode, target_type, transform, target_transform, transforms)
    
    def color2trainid(self, target):
        height = target.shape[1]
        weight = target.shape[2]
        id_map = np.zeros((height, weight))
        # TODO fix label
        for h in range(height):
            for w in range(weight):
                color = tuple(target[:, h, w])
                print(color)
                try:
                    trainid = self.color2index[color]
                    id_map[h, w] = trainid
                except:
                    id_map[h, w] = 0
        return  flow.tensor(id_map, dtype=flow.long) 
        
        
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        print("---------------")
        print(target.shape)
        print("---------------")
        target = self.color2trainid(target)
        
        data_sample = Instance(
            images=DistTensorData(image, placement_idx=0),
            labels=DistTensorData(target, placement_idx=-1),
        )
        
        return data_sample
