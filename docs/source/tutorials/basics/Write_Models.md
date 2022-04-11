# Write Models

This section introduces how to implement a new model entirely from scratch and make it compatible with LiBai.


## Construct Models in LiBai

LiBai uses [LazyConfig](https://libai.readthedocs.io/en/latest/tutorials/Config_System.html) for a more flexible config system, which means you can simply import your own model in your config and train it under LiBai.

For image classification task, the input data is usually a batch of images and labels. The following code shows how to build a toy model for this task. Import in your code:
```python
# toy_model.py
import oneflow as flow
import oneflow.nn as nn


class ToyModel(nn.Module):
    def __init__(self, 
        num_classes=1000, 
    ):
        super().__init__()
        self.features = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, images, labels=None):
        x = self.features(images)
        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.classifier(x)

        if labels is not None and self.training:
            losses = self.loss_func(x, labels)
            return {"losses": losses}
        else:
            return {"prediction_scores": x}
```

**Note:**
- For classification models, the ``forward`` function must have ``images`` and ``labels`` as arguments, which correspond to the output in ``__getitem__`` of LiBai's built-in datasets. Please refer to [imagenet.py](https://github.com/Oneflow-Inc/libai/blob/main/libai/data/datasets/imagenet.py) for more details about the dataset.
- **This toy model** will return ``losses`` during training and ``prediction_scores`` during inference, and both of them should be the type of ``dict``, which means you should implement the ``loss function`` in your model, like ``self.loss_func=nn.CrossEntropyLoss()`` as the ToyModel shows above.


## Import the model in config

With ``LazyConfig System``, you can simply import the model in your config file. The following code shows how to use ``ToyModel`` in your config file:
```python
# config.py
from libai.config import LazyCall
from toy_model import ToyModel

model = LazyCall(ToyModel)(
    num_classes=1000
)
```



