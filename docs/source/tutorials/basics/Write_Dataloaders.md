# Write Dataloaders

This tutorial explains how the dataset APIs work, and how to customize your own datasets with them.

## Build Common Dataloaders 

To build dataloaders in LiBai, we highly recommend users to use the default `build_nlp_train_val_test_loader`, `build_nlp_train_loader`, `build_nlp_test_loader`, `build_image_train_loader` and `build_image_test_loader` which are defined in [`libai/data/build.py`](https://github.com/Oneflow-Inc/libai/blob/main/libai/data/build.py) for most of the common cases.

The only thing you need to do is to write pytorch style `Dataset`, and return `Instance` structure in `__getitem__`. The `Instance` structure stores the attributes of an instance (e.g., image, tokens) as "fields", and the `DistTensorData` structure provides a standard `to_global()`(called in `get_batch()`) function to convert local tensors to global tensors.

The returned instance by `__getitem__` function must contain the same keys with the `args` passed in `forward` function of the `model`. The following shows an example:

**NOTE:** Set `placement_idx=-1` in `DistTensorData` when the `tensor` is **only** used in `loss_function`, it is used for pipeline parallel training.

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
        lm_labels = flow.tensor(text[1:2], dtype=flow.long)
        # attention_mask must be a [0, 1] metric
        attention_mask = flow.tensor(text[2:3], dtype=flow.long)
        loss_mask = flow.tensor(text[3:], dtype=flow.long)
        # the keys (`input_ids` ... `labels`) should be same as the parameter name of model.forward()
        sample = Instance(
            input_ids=DistTensorData(input_ids),
            # attention_mask must be a [0, 1] metric
            attention_mask=DistTensorData(attention_mask),
            loss_mask=DistTensorData(lm_labels, placement_idx=-1),
            labels=DistTensorData(lm_labels, placement_idx=-1),
        )
        return sample

# my_model.py
import oneflow.nn as nn

class MyModel(nn.Module):
    ...
    
    # the parameters' name is the same as the returned key in __getitem__
    def forward(self, input_ids, attention_mask, loss_mask, labels):
        ...
```

In particular, the values of `attention_mask` can only be `0` or `1` if you need to generate your own `attention_mask`. Because LiBai has already processed `attention_mask` in [`libai/layers/attention.py`](https://github.com/Oneflow-Inc/libai/blob/main/libai/layers/attention.py) as follows:

```python
attention_scores = flow.mul(attention_scores, attention_mask)
attention_scores = attention_scores - 10000.0 * (1 - attention_mask)
attention_weights = flow.softmax(attention_scores, dim=-1)
```

After finishing your `MyDataset`, set `dataloader` in your `config.py` depending on your needs. If you have only one training dataset for nlp task and want to split it into `train`, `valid` and `test` datasets automatically, you can choose `build_nlp_train_val_test_loader`, the evaluation will be calculated in `valid` and `test` dataset. 

Otherwise, you can choose `build_nlp_train_loader` && `build_nlp_test_loader` or  `build_image_train_loader` && `build_image_test_loader` in `config.py` according to your own needs.
see [`libai/data/build.py`](https://github.com/Oneflow-Inc/libai/blob/main/libai/data/build.py) for more details.