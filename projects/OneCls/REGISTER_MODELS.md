## Register Your Own Models in OneCls
Here is the tutorial which introduces how to write your own models to train under LiBai for classification tasks.

### Step-1: write your own model
Here we use a toy model as an example:
```python
class ToyModel(nn.Module):
    def __init__(self, 
        num_classes=1000, 
    ):
        super().__init__()
        self.features = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.classifier(x)
        return x
```

### Step-2: register the model into flowvision
```python
class ToyModel(nn.Module):
    def __init__():
        ...
    
    def forward(self, x):
        ...

from flowvision.models import ModelCreator

# register the model create function into flowvision
@ModelCreator.register_model
def toy_model(pretrained=False, progress=True, **kwargs):
    return ToyModel()
```

### Step-3: import your model in config file
Please use **absolute path** here:
```python
# config.py
from projects.OneCls.modeling.example import toy_model
```

### Step-4: update `model.model_name` in config
```python
# config.py
from projects.OneCls.modeling.example import toy_model
...

# Add model for training
model = LazyCall(VisionModel)(
    model_name = "toy_model",  # call toy_model
    pretrained = False,
    num_classes = 1000,
    loss_func = LazyCall(SoftTargetCrossEntropy)()
)
```

### Addition

We provide the whole example code here:
- [example model](./modeling/example.py)
- [example config](./configs/example.py)

Please check the files for more details.

## Training your own model in LiBai
After finishing all the above steps, you can train your own model under LiBai like:
```python
cd /path/to/libai
bash tools/train.sh projects/OneCls/train_net.py projects/OneCls/configs/config.py 8
```