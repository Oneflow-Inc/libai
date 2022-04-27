import oneflow as flow
import flowvision as vision

class CIFAR_Dataset(flow.utils.data.Dataset):
    def __init__(self,root,train,download,transform) -> None:
        super().__init__()
        self.data = vision.datasets.CIFAR10(root=root, train=train,
                                                download=download, transform=transform)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index]
