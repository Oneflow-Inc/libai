


class ImageNetDataset(datasets.ImageFolder):
    """ImageNet Dataset
    """

    def __init__(self, 
                 root: str,
                 train: bool = True, 
                 transform: Optional[Callable] = None,
                 **kwargs):
        prefix = "train" if train else "val"
        pass