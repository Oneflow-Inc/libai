# Write Models

Here we provides the tutorials for users who want to implement new model entirely from scratch and make it compatible in LiBai.


## Construct Models in LiBai

LiBai uses ``LazyConfig`` for more flexible config system, which means you can simply import your own model in your config and train it under LiBai.

For example, in image classification task, the input data is usually a batch of images with its targets during training and only input a batch of images when testing, to build a toy model for image classification task, import this code in your own code:
```python
# toy_model.py
import oneflow as flow
import oneflow.nn as nn


class ToyModel(nn.Module):
    def __init__(self, 
        num_classes=1000, 
        loss_func=None
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
- For classification models, the ``forward`` function of each model must have ``images`` and ``labels`` as arguments, which corresponds to the output of LiBai's built-in datasets, please refer to [imagenet.py](https://github.com/Oneflow-Inc/libai/blob/main/libai/data/datasets/imagenet.py) for more details about the dataset.
- Each model returns ``losses`` during training and returns ``prediction_scores`` during inference, and both of them should be the type of ``dict``, which means you should implement the ``loss function`` in your model, like ``self.loss_func=nn.CrossEntropyLoss()`` as the ToyModel.


## Import the model in config

With ``LazyConfig System``, you can simply import the model in your config file to use ``ToyModel`` for training as follows:
```python
# config.py
from libai.config import LazyCall
from toy_model import ToyModel

model = LazyCall(ToyModel)(
    num_classes=1000
)
```



