import oneflow as flow
import random
from PIL import ImageFilter, ImageOps
from .common.models.graph import graph
from .common.models.moco_v3.MoCo_v3_Vit import model
from .common.train import train
from .common.optim import optim
from .common.data.imagenet import dataloader
from flowvision import transforms

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/dataset/imagenet/extract"
dataloader.test[0].dataset.root = "/dataset/imagenet/extract"


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


# follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
augmentation1 = [
    transforms.RandomResizedCrop(224, scale=(.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    ], p=0.8),
    # transforms.RandomGrayscale(p=0.2), # oneflow does not support RandomGrayscale
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]

augmentation2 = [
    transforms.RandomResizedCrop(224, scale=(.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    ], p=0.8),
    # transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0), 
    transforms.RandomApply([Solarize()], p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        # return [im1, im2]
        return flow.cat((im1, im2), dim=0)

dataloader.train.mixup_func = None

# Add augmentation Func
dataloader.train.dataset[0].transform=TwoCropsTransform(
                                               transforms.Compose(augmentation1),
                                               transforms.Compose(augmentation2))



# Refine optimizer cfg for moco v3 model
optim.lr = 1.5e-4
optim.eps = 1e-8
optim.weight_decay = .1

# Refine train cfg for moco v3 model
train.train_micro_batch_size = 128  # 128
train.test_micro_batch_size = 128  # 128
train.train_epoch = 300
train.warmup_ratio = 40 / 300
train.eval_period = 1 # 1000
train.log_period = 1

# Scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 1.5e-4
train.scheduler.warmup_method = "linear"

# Set fp16 ON
train.amp.enabled = True

graph.enabled = False