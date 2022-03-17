# Distributed Configuration
In LiBai, you can try out different parallel modes easily by updating the [training config file](https://github.com/Oneflow-Inc/libai/blob/main/configs/common/train.py).
```python
dist=dict(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
)
```
For example, you can set `data_parallel_size=2` to split the input data into two groups for data parallel training.

**Examples:**
On the above table, **BERT** model supports three parallelism techniques, if we have 1 node with 8 GPUs, you can try out different combinations of parallelism training techniques by updating [bert config file](../../../configs/bert_large_pretrain.py) as follows:
- **Pure Data Parallel Training on 8 GPUs**

In this example, the input data will be splitted into 8 parts on batch dim for data parallel training on 8 GPUs.
```python
from .common.train import train
...

train.dist.data_parallel_size = 8
```

- **Pure Tensor Parallel Training on 8 GPUs**

In this example, the weight of the layers in the model be splitted into 8 parts for tensor parallel training on 8 GPUs.
```python
from .common.train import train
...

train.dist.tensor_parallel_size = 8
```

- **Pure Pipeline Parallel Training on 8 GPUs**

In this example, 8 GPUs will be splitted into 8 stages, different layers of the model will be put on different stages automatically for pipeline parallel training.
```python
from .common.train import train
...

train.dist.pipeline_parallel_size = 8
```

- **Data Parallel + Tensor Parallel for 2D Parallel Training on 8 GPUs**

In this example, 8 GPUs will be splitted into **2 groups**, each group contains **4 GPUs**, and the input data will be splitted into 2 parts on batch dim for data parallel training between 2 groups. And in each group, the weight of the layers in the model will be splited into 4 parts on 4 GPUs for tensor parallel training.
```python
from .common.train import train
...

train.dist.data_parallel_size = 2
train.dist.tensor_parallel_size = 4
```

- **Data Parallel + Pipeline Parallel for 2D Parallel Training on 8 GPUs**

In this example, 8 GPUs will be splitted into **2 groups**, each group contains **4 GPUs**, and the input data will be splitted into 2 parts on batch dim for data parallel training. And each group contains **4 stages**, different layers in the model will be put on different stages automatically for pipeline parallel training.
```python
from .common.train import train
...

train.dist.data_parallel_size = 2
train.dist.pipeline_parallel_size = 4
```

- **Tensor Parallel + Pipeline Parallel for 2D Parallel Training on 8 GPUs**

In this example, 8 GPUs will be splitted into **2 groups**, each group contains **4 GPUs**, and the weight of the layers in the model be splitted into 2 parts for tensor parallel training between 2 groups. And each group contains **4 stages**, different layers in the model will be put on different stages automatically for pipeline parallel training.
```python
from .common.train import train
...

train.dist.tensor_parallel_size = 2
train.dist.pipeline_parallel_size = 4
```

- **Data Parallel + Tensor Parallel + Pipeline Parallel for 3D Parallel Training on 8 GPUs**

In this example, 8 GPUs will also be splitted into **2 groups**, but each group will also be splitted into **2 mini groups**, each mini groups contains 2 GPUs, the input data will be splitted into two parts on batch dim for data parallel training on 2 groups, and in each group, the weight of the layers in the model will be splitted into 2 parts for tensor parallel training on **2 mini groups**, and each mini group contrain **2 stages**, different layers in the model will be put on different stages for pipeline parallel training.
```python
from .common.train import train
...

train.dist.data_parallel_size = 2
train.dist.tensor_parallel_size = 2
train.dist.pipeline_parallel_size = 2
```

You can also use **command line** to control the parallelization mode as follows:
```bash
bash tools/train.sh tools/train_net.py configs/bert_large_pretrain.py \
8 \  # num of gpus
train.dist.data_parallel_size=2 \
train.dist.tensor_parallel_size=2 \
train.dist.pipeline_parallel_size=2 \
```