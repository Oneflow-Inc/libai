import pdb
import oneflow as flow
import flowvision as vision
from libai.data.structures import DistTensorData, Instance

class CIFAR_Dataset(flow.utils.data.Dataset):
    def __init__(self,root,train,download,transform) -> None:
        super().__init__()
        self.data = vision.datasets.CIFAR10(root=root, train=train,
                                                download=download, transform=transform)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        tensors = {}
        d = self.data[index]
        inputs = flow.tensor(d[0],dtype=flow.float)
        inputs = inputs.view(-1)
        labels = flow.tensor(d[1],dtype=flow.long)
        tensors["x"] = DistTensorData(inputs,placement_idx=-1)
        tensors["labels"] = DistTensorData(labels)
        return Instance(**tensors)

    # def __getitem__(self, i):
    #     feature = self.features[i]
    #     tensors = {}
    #     for k, v in feature.__dict__.items():
    #         if v is not None:
    #             if k == "labels":
    #                 dtype = flow.long if isinstance(v, int) else flow.float
    #                 t = flow.tensor(v, dtype=dtype)
    #                 tensors[k] = DistTensorData(t, placement_idx=-1)
    #             else:
    #                 t = flow.tensor(v, dtype=flow.long)
    #                 tensors[k] = DistTensorData(t)
    #     sample = Instance(**tensors)
    #     return sample