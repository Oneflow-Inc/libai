import pdb
from projects.MOE.dataset.dataset import CIFAR_Dataset
import flowvision as vision
from oneflow.utils.data import DataLoader
from libai.data.structures import DistTensorData, Instance

transform = vision.transforms.Compose(
    [vision.transforms.ToTensor(),
        vision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data_root = "./projects/MOE/data"

def trivial_batch_collator(batch):
    assert isinstance(batch[0], Instance), "batch[0] must be `instance` for trivial batch collator"
    batch = Instance.stack(batch)
    return batch

dataset = CIFAR_Dataset(data_root,True,True,transform=transform)
dataset[1]

dataloader = DataLoader(
        dataset,
        collate_fn=trivial_batch_collator,
        batch_size=4,
    )

for d in dataloader:
    pdb.set_trace()
