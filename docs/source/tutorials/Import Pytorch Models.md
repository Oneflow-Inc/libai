# Import PyTorch models to Libai

`oneflow.nn.Module` and `torch.nn.Module` implement the same interface, so you can simply replace `torch` in the model definition with `oneflow` to embed the existing model structure into Libai.

Take [MobileNetV2](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py) training cifar100 as an example.

### Replace `torch` with `oneflow`

libai/model/MobileNetV2.py (original code)

```python
"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math

__all__ = ['mobilenetv2']

... # omit some code

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)
```

It is only necessary to **change the `torch` in the import statement to `oneflow`** to convert it into an oneflow model while ensuring the model structure is identical. Then add the training loss calculation at the end of the model to form a complete training model, i.e.

libai/model/MobileNetV2.py

```python
"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import oneflow.nn as nn
import math

__all__ = ['mobilenetv2']

... # omit some code

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)

class MobileNetV2Training(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = MobileNetV2(**kwargs)
        self.loss = oneflow.nn.CrossEntropyLoss()
    
    def forward(self, images, labels):
        logits = self.model(images)
        if labels is not None and self.training:
            loss = self.loss(logits, labels.long())
            return {"loss": loss}
        else:
            return {"prediction_scores": logits}
```

Now we have obtained a MobileNetV2 model that is usable under Libai.

### Set the training configuration

Create a configuration file: configs/cifar.py

```python
from libai.config import LazyCall
from libai.models.MobileNetV2 import MobileNetV2Training
from .common.models.graph import graph
from .common.train import train
from .common.optim import optim
from .common.data.cifar100 import dataloader

model = LazyCall(MobileNetV2Training)()
```

### Modify the configuration of the cifar100 dataset
configs/common/data/cifar100.py (modify the `mixup_func` in dataloader.train to `None`)

```python
dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_image_train_loader)(
    dataset=[
        LazyCall(CIFAR10Dataset)(
            root="./",
            train=True,
            download=True,
            transform=train_aug,
        ),
    ],
    num_workers=12,
    mixup_func=None
)
```

### Start training

The imported PyTorch model structure can be used for training by running `zsh tools/train.sh tools/train_net.py configs/cifar.py 1` in the Libai directory.


