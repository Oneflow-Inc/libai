# Distributed Configuration

In LiBai, you can try out different parallel modes easily by updating the distributed config in [training config file](https://github.com/Oneflow-Inc/libai/blob/main/configs/common/train.py).
```python
# Distributed arguments
dist=dict(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
)
```
For example, you can set `data_parallel_size=2` which will automatically split the input data into two groups for data parallel training.

## Distributed Setting Example
Here we provide simple examples for users to understand the basic configuration of LiBai's distributed settings. LiBai's **BERT** model supports three parallelism techniques, and here we use 1 node with 8 GPUs as an example. If you do not change any default settings, LiBai will execute **data parallel training as default**. You can try out different combinations of parallelism training techniques by updating [bert config file](../../../configs/bert_large_pretrain.py) as follows:
#### **Pure Data Parallel Training on 8 GPUs**

In this example, the input data will be splitted into 8 parts on batch dim for data parallel training on 8 GPUs.
```python
from .common.train import train
...

train.dist.data_parallel_size = 8
```

#### **Pure Tensor Parallel Training on 8 GPUs**

In this example, the weight of the layers in the model be splitted into 8 parts for tensor parallel training on 8 GPUs.
```python
from .common.train import train
...

train.dist.tensor_parallel_size = 8
```

#### **Pure Pipeline Parallel Training on 8 GPUs**

In this example, 8 GPUs will be splitted into 8 stages, different layers of the model will be put on different stages automatically for pipeline parallel training.
```python
from .common.train import train
...

train.dist.pipeline_parallel_size = 8
```

#### **Data Parallel + Tensor Parallel for 2D Parallel Training on 8 GPUs**

In this example, 8 GPUs will be splitted into **2 groups**, each group contains **4 GPUs**, and the input data will be splitted into 2 parts on batch dim for data parallel training between 2 groups. And in each group, the weight of the layers in the model will be splited into 4 parts on 4 GPUs for tensor parallel training.
```python
from .common.train import train
...

train.dist.data_parallel_size = 2
train.dist.tensor_parallel_size = 4
```

#### **Data Parallel + Pipeline Parallel for 2D Parallel Training on 8 GPUs**

In this example, 8 GPUs will be splitted into **2 groups**, each group contains **4 GPUs**, and the input data will be splitted into 2 parts on batch dim for data parallel training. And each group contains **4 stages**, different layers in the model will be put on different stages automatically for pipeline parallel training.
```python
from .common.train import train
...

train.dist.data_parallel_size = 2
train.dist.pipeline_parallel_size = 4
```

#### **Tensor Parallel + Pipeline Parallel for 2D Parallel Training on 8 GPUs**

In this example, 8 GPUs will be splitted into **2 groups**, each group contains **4 GPUs**, and the weight of the layers in the model be splitted into 2 parts for tensor parallel training between 2 groups. And each group contains **4 stages**, different layers in the model will be put on different stages automatically for pipeline parallel training.
```python
from .common.train import train
...

train.dist.tensor_parallel_size = 2
train.dist.pipeline_parallel_size = 4
```

#### **Data Parallel + Tensor Parallel + Pipeline Parallel for 3D Parallel Training on 8 GPUs**

In this example, 8 GPUs will also be splitted into **2 groups**, but each group will also be splitted into **2 mini groups**, each mini groups contains 2 GPUs, the input data will be splitted into two parts on batch dim for data parallel training on 2 groups, and in each group, the weight of the layers in the model will be splitted into 2 parts for tensor parallel training on **2 mini groups**, and each mini group contrain **2 stages**, different layers in the model will be put on different stages for pipeline parallel training.
```python
from .common.train import train
...

train.dist.data_parallel_size = 2
train.dist.tensor_parallel_size = 2
train.dist.pipeline_parallel_size = 2
```

**Note:** `train.dist.data_parallel_size` will be automatically calculated by `(gpu_nums / (tensor_parallel_size * pipeline_parallel_size))` if only `train.dist.tensor_parallel_size` and `train.dist.pipeline_parallel_size` are setted, for example:
```python
from .common.train import train
...
# only set tensor_parallel_size and pipeline_parallel_size
train.dist.tensor_parallel_size = 2
train.dist.pipeline_parallel_size = 2
```
And the `data_parallel_size` will be automatically setted to `(8 / (2 * 2)) = 2`


## Update Distributed Config with Command Line
You can also use **command line** to control the parallelization mode as follows:
```bash
bash tools/train.sh tools/train_net.py configs/bert_large_pretrain.py \
8 \  # num of gpus
train.dist.data_parallel_size=2 \
train.dist.tensor_parallel_size=2 \
train.dist.pipeline_parallel_size=2 \
```