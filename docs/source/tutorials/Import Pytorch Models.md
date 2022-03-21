# Train PyTorch models with LiBai seamlessly

`oneflow.nn.Module` implements the same interface as `torch.nn.Module`, so it's easy for users to convert a pytorch model to oneflow model and train it based on LiBai.

Take [MobileNetV2](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py) training cifar100 as an example.

### Steps for converting `torch` model script to `oneflow`

Take MobileNetV2 as an example, the original implementation can be found in [there](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py).


To convert the above script into oneflow, the only thing user need to do is simply changing import torch.nn as nn to import oneflow.nn as nn. Then add the loss module at the end of the forward function to form a complete training module.

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

Now we have created a MobileNetV2 model which is usable in Libai.

### Create configuration file

For example configs/mobilenetv2_cifar100.py

```python
from libai.config import LazyCall
from libai.models.MobileNetV2 import MobileNetV2Training
from .common.models.graph import graph
from .common.train import train
from .common.optim import optim
from .common.data.cifar100 import dataloader

model = LazyCall(MobileNetV2Training)()
dataloader.train.mixup_func = None  # remove mixup augmentation
```

### Start training

After the above preparation, users can start training on single GPU by running `bash tools/train.sh tools/train_net.py configs/cifar.py 1` under Libai.


