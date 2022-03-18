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
Here we provide simple examples for users to understand the basic configuration of LiBai's distributed settings. LiBai's **BERT** model supports three parallelism techniques (**data parallel training**, **tensor parallel training** and **pipeline parallel training**), and here we use 1 node with 8 GPUs as an example. If you do not change any default settings, LiBai will execute **data parallel training** by default. You can try out different combinations of parallelism training techniques by updating [bert config file](https://github.com/Oneflow-Inc/libai/blob/main/configs/bert_large_pretrain.py) as follows:
#### **Pure Data Parallel Training on 8 GPUs**

In this example, the model is replicated on 8 GPUs, and each replica handles only part of the input data during iteration.
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
**Note:** For models which have been configured with pipeline parallelism, e.g., BERT, GPT-2, T5 and VisionTransformer model in LiBai, you can simply update the distributed config to excute pipeline parallel training on these models. If you need to train your own model with pipeline parallel strategy, please refer to [Write Models]() for more details about configuring your own model with pipeline parallelism.

#### **Data Parallel + Tensor Parallel for 2D Parallel Training on 8 GPUs**

In this example, 8 GPUs will be splitted into **2 groups**, each group contains **4 GPUs**, and the input data will be splitted into 2 parts by chunking in the batch dimension for data parallel training between 2 groups. The model is replicated between **2 data parellel groups**, within each group, the weight of each layers will be splited across **4 GPUs** for tensor parallel training.

```python
from .common.train import train
...

train.dist.data_parallel_size = 2
train.dist.tensor_parallel_size = 4
```

#### **Data Parallel + Pipeline Parallel for 2D Parallel Training on 8 GPUs**

In this example, 8 GPUs will be splitted into **2 groups**, each group contains **4 GPUs**, and the input data will be splitted into 2 parts by chunking in the batch dimension for data parallel training. The model is replicated between **2 data parellel groups**, and each group contains **4 stages**, different layers in the model will be put on different stages automatically for pipeline parallel training.
```python
from .common.train import train
...

train.dist.data_parallel_size = 2
train.dist.pipeline_parallel_size = 4
```

#### **Tensor Parallel + Pipeline Parallel for 2D Parallel Training on 8 GPUs**

In this example, 8 GPUs will be splitted into **4 stages**, each stage contains **2 GPUs** as a **group**. And different layers in the model will be put on different stages automatically for pipeline parallel training. The weight of the layers be put on the specific stage will be splitted into 2 parts for tensor parallel training within the group. 

```python
from .common.train import train
...

train.dist.tensor_parallel_size = 2
train.dist.pipeline_parallel_size = 4
```

#### **Data Parallel + Tensor Parallel + Pipeline Parallel for 3D Parallel Training on 8 GPUs**

In this example, 8 GPUs will also be splitted into **2 stages**, different layers in the model will be put on different stages for pipeline parallel training. Each stage only contains a portion of the whole model, and each stage will be splitted into **2 groups**. In each stage, the model is replicated between **2 data parellel groups**, each **data parallel group** contains **2 GPUs**, the input data will be splitted into 2 parts by chunking in the batch dimension for data parallel training between **2 data parallel groups**, within each group, the weight of each layers will be splited across **2 GPUs** for tensor parallel training.

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
You can also control the parallelization strategy by **command line** paremeters as follows:

```bash
bash tools/train.sh tools/train_net.py configs/bert_large_pretrain.py \
8 \  # num of gpus
train.dist.data_parallel_size=2 \
train.dist.tensor_parallel_size=2 \
train.dist.pipeline_parallel_size=2 \
```