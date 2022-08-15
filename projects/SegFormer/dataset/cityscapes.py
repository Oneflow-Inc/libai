import flowvision
from PIL import Image

from libai.data.structures import DistTensorData, Instance


class CityScapes(flowvision.datasets.Cityscapes):
           
    def __init__(self, root, split = "train", mode = "fine", target_type = "instance", transform = None, target_transform = None, transforms = None):
        super().__init__(root, split, mode, target_type, transform, target_transform, transforms)
       
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert("RGB")

        targets = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]
        
   
        if self.transforms is not None:
            image, target = self.transforms(image, target)
                
        data_sample = Instance(
            images=DistTensorData(image, placement_idx=0),
            labels=DistTensorData(target.long(), placement_idx=-1),
        )
        
        return data_sample
    
    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == "instance":
            return f"{mode}_instanceIds.png"
        elif target_type == "semantic":
            return f"{mode}_labelTrainids.png"
        elif target_type == "color":
            return f"{mode}_color.png"
        else:
            return f"{mode}_polygons.json"
