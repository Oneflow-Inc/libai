from libai.data.datasets.imagenet import ImageNetDataset
from libai.data.structures import Instance

class PretrainingImageNetDataset(ImageNetDataset):
    """ImageNet Dataset in LiBai for Pretraining
    Return:
        images: ImageNet train set images
    """
    def __getitem__(self, index: int):
        data_sample = super().__getitem__(index)
        return Instance(
            images = data_sample.get("images")
        )