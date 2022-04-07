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

In particular, if you need to generate your own `attention_mask`, the `attention_mask` must be a [0, 1] metric. Cause LiBai has already process `attention_mask` in `libai/layers/attention.py`

```python
class MultiheadAttention(nn.Module):
    ...

    def forward(
        self,
        hidden_states: flow.Tensor,
        encoder_states: flow.Tensor = None,
        attention_mask: flow.Tensor = None,
        past_key_value: Tuple[flow.Tensor, flow.Tensor] = None,
        use_cache: bool = False,
    ):
        ...

        attention_scores = flow.matmul(query, key, transpose_b=True, alpha=self.norm_factor)

        # your passed attention_mask
        if attention_mask is not None:
            if self.scale_mask_softmax_fusion:
                attention_weights = flow._C.fused_scale_mask_softmax(
                    attention_scores, attention_mask, fill_value=-10000.0
                )
            else:
                if self.coeff is not None:
                    attention_scores *= self.coeff
                attention_scores = flow.mul(attention_scores, attention_mask)
                attention_scores = attention_scores - 10000.0 * (1 - attention_mask)
                attention_weights = flow.softmax(attention_scores, dim=-1)
        else:
            attention_weights = flow.softmax(attention_scores, dim=-1)

        attention_weights = self.dropout(attention_weights)
        context = flow.matmul(attention_weights, value)

```

After finish your `MyDataset`, set `dataloader` in your `config.py` according to your own needs. If you only have one training dataset for nlp task and you want to split it to `train`, `valid` and `test` dataset automatically, you can choose `build_nlp_train_val_test_loader`, the evaluation will be calculated in `valid` and `test` dataset. 

Otherwise, you can choose `build_nlp_train_loader` && `build_nlp_test_loader` or  `build_image_train_loader` && `build_image_test_loader` in `config.py` according to your own needs.
see [`libai/data/build.py`](https://github.com/Oneflow-Inc/libai/blob/main/libai/data/build.py) for more details.