# Write Dataloaders

In this section, we will introduce how to implement a custom Dataloader in LiBai.

# Build Common Dataloaders 

In most cases, we highly recommend you use default `build_nlp_train_val_test_loader`, `build_nlp_train_loader`, `build_nlp_test_loader`, `build_image_train_loader` and `build_image_test_loader` defined in [`libai/data/build.py`](https://github.com/Oneflow-Inc/libai/blob/main/libai/data/build.py) to build dataloaders in LiBai.

The only thing you should do is writing `Dataset` like torch, and return `Instance` structure in `__getitem__`. In `__getitem__` function, the `key` returned by the method must be consistent with the parameter name of the `forward` function in the `model`. Here is an example code: 

> NOTE: Set `placement_idx=-1` in `DistTensorData` when the `tensor` is **only** used in `loss_function`, it is used for pipeline parallel training.

```python
# my_dataset.py
import numpy as np
import oneflow as flow

from libai.data.structures import DistTensorData, Instance

class MyDataset(flow.utils.data.Dataset):

    ...

    def __getitem__(self, idx):
        text = np.array(self.dataset[idx], dtype=np.long)
        # transfer to flow.tensor
        input_ids = flow.tensor(text[:-1], dtype=flow.long)
        lm_labels = flow.tensor(text[1:], dtype=flow.long)
        # the keys (`input_ids` and `labels`) should be same as the parameter name of model.forward()
        sample = Instance(
            input_ids=DistTensorData(input_ids),
            labels=DistTensorData(lm_labels, placement_idx=-1),
        )
        return sample

# my_model.py
import oneflow.nn as nn

class MyModel(nn.Module):
    ...
    
    # the parameters' name is the same as the returned key in __getitem__
    def forward(self, input_ids, labels):
        ...
```
