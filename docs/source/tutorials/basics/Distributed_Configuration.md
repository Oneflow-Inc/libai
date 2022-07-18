# Distributed Configuration

In LiBai, you can try out different parallel strategies by simply changing the distributed config in [training config file](https://github.com/Oneflow-Inc/libai/blob/main/configs/common/train.py).
```python
# Distributed arguments
dist=dict(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,

        # users must set the `pipeline_num_layers` attribute when `pipeline_parallel_size > 1`
        pipeline_num_layers=None,
        # users could customize the number of layers in different stages
        # by setting the `custom_pipeline_stage_id ` attribute which is used for
        # manually balance calculation between stages when running pipeline parallelism
        # e.g. you can set `custom_pipeline_stage_id=[0, 0, 0, 1]`
        # for `pipeline_num_layers=4 and pipeline_parallel_size=2`
        # which means the first 3 layers will be placed on stage0 and
        # the last layer will be placed on stage1
        # NOTE: if it is None, LiBai will automatically set pipeline_stage_id
        # `auto_pipeline_stage_id` and `actual_pipeline_stage_id` will be saved in `config.yaml`
        custom_pipeline_stage_id=None,
)
```
For example, you can set `data_parallel_size=2` which will automatically split the input data into two groups for data parallel training.

## Distributed Setting Example
Here are some simple examples for you to understand the basic configuration of LiBai's distributed settings. LiBai's **BERT** model supports three parallelism techniques (**data parallel training**, **tensor parallel training** and **pipeline parallel training**). Take 1 node with 8 GPUs as an example. If you do not change any default settings, LiBai will execute **data parallel training** by default. You can try out different combinations of parallelism training techniques by updating [bert config file](https://github.com/Oneflow-Inc/libai/blob/main/configs/bert_large_pretrain.py) as follows:

#### **Pure Data Parallel Training on 8 GPUs**

In this example, the model is replicated on 8 GPUs, and each replica handles only part of the input data during iteration.
```python
from .common.train import train
...

train.dist.data_parallel_size = 8
```

#### **Pure Tensor Parallel Training on 8 GPUs**

In this example, the weight of the layers in the model will be split into 8 parts for tensor parallel training on 8 GPUs.
```python
from .common.train import train
...

train.dist.tensor_parallel_size = 8
```

**Note:** This only works for models built with ``libai.layers``.

#### **Pure Pipeline Parallel Training on 8 GPUs**

In this example, 8 GPUs will be split into 8 stages, and different layers of the model will be put on different stages automatically for pipeline parallel training.
```python
from .common.train import train
...

train.dist.pipeline_parallel_size = 8

train.dist.pipeline_num_layers = model.cfg.hidden_layers
```

**Note:** 
- `train.dist.pipeline_num_layers` must be set consistent with the model layers. If unset, it will use the default value `1000`,
which might trigger unexpected behavior.

- For models which have been configured with pipeline parallelism(e.g., BERT, GPT-2, T5 and ViT), you can simply update the distributed config to execute pipeline parallel training on them. If you need to train your own model with pipeline parallel strategy, please refer to [Write Models](https://libai.readthedocs.io/en/latest/tutorials/basics/Write_Models.html) for more details about configuring your own model with pipeline parallelism.

#### **Data Parallel + Tensor Parallel for 2D Parallel Training on 8 GPUs**

In this example, 8 GPUs will be split into **2 groups**, and each group contains **4 GPUs**. The input data will be split into 2 parts by chunking in the batch dimension for data parallel training between 2 groups. The model is replicated between **2 data-parellel groups**. Within each group, the weight of each layers will be splited across **4 GPUs** for tensor parallel training.

```python
from .common.train import train
...

train.dist.data_parallel_size = 2
train.dist.tensor_parallel_size = 4
```
Here we provide a specific example for you to understand this. We number 8 GPUs from 0 to 7, e.g., ``[0, 1, 2, 3, 4, 5, 6, 7]``, and for ``data parallel + tensor parallel``, 8 GPUs will be split into 2 groups as ``[[0, 1, 2, 3], [4, 5, 6, 7]]``, ``GPU: [0, 1, 2, 3]`` as group-0 and ``GPU: [4, 5, 6, 7]`` as group-1. The model is replicated between group-0 and group-1. In group-0, the model will execute tensor parallel between ``GPU: [0, 1, 2, 3]``. In group-1, the model will execute tensor parallel between ``GPU: [4, 5, 6, 7]``, and each group only handle a portion of the input data for data parallel training.

#### **Data Parallel + Pipeline Parallel for 2D Parallel Training on 8 GPUs**

In this example, 8 GPUs will be split into **4 stages**. Each stage contains **2 GPUs** which will be split into **2 data-parallel groups**. Each stage only contains a portion of the model. The weight of the layers put on the specific stage is replicated on **2 data-parallel groups**. Each group handles a portion of the input data.
```python
from .common.train import train
...

train.dist.data_parallel_size = 2
train.dist.pipeline_parallel_size = 4

train.dist.pipeline_num_layers = model.cfg.hidden_layers
```

#### **Tensor Parallel + Pipeline Parallel for 2D Parallel Training on 8 GPUs**

In this example, 8 GPUs will be split into **4 stages**, each stage contains **2 GPUs** as a **group**. And different layers in the model will be put on different stages automatically for pipeline parallel training. The weight of the layers put on the specific stage will be split into 2 parts for tensor parallel training within the group. 

```python
from .common.train import train
...

train.dist.tensor_parallel_size = 2
train.dist.pipeline_parallel_size = 4

train.dist.pipeline_num_layers = model.cfg.hidden_layers
```

#### **Data Parallel + Tensor Parallel + Pipeline Parallel for 3D Parallel Training on 8 GPUs**

In this example, 8 GPUs will also be split into **2 stages**, and different layers in the model will be put on different stages for pipeline parallel training. Each stage only contains a portion of the whole model, and each stage will be split into **2 groups**. In each stage, the model will be replicated between **2 data-parallel groups**, and each **data-parallel group** contains **2 GPUs**. The input data will be split into 2 parts by chunking in the batch dimension for data-parallel training between **2 data-parallel groups**. Within each group, the weight of each layer will be split across **2 GPUs** for tensor parallel training.

```python
from .common.train import train
...

train.dist.data_parallel_size = 2
train.dist.tensor_parallel_size = 2
train.dist.pipeline_parallel_size = 2

train.dist.pipeline_num_layers = model.cfg.hidden_layers
```


**Note:** `train.dist.data_parallel_size` will be automatically calculated by `(gpu_nums / (tensor_parallel_size * pipeline_parallel_size))` if only `train.dist.tensor_parallel_size` and `train.dist.pipeline_parallel_size` are set. For example:

```python
from .common.train import train
...
# only set tensor_parallel_size and pipeline_parallel_size
train.dist.tensor_parallel_size = 2
train.dist.pipeline_parallel_size = 2

train.dist.pipeline_num_layers = model.cfg.hidden_layers
```
And the `data_parallel_size` will be automatically set to `(8 / (2 * 2)) = 2`


#### **Set `custom_pipeline_stage_id` for Load Balance**
In most cases, the transformer layers of common models have the same computational overhead, so there is no need to set `custom_pipeline_stage_id`.

But when transformer layers have unbalanced computational overhead, you can set `custom_pipeline_stage_id` for manually balance the compuation between stages in pipeline_parallelism

For example:
```python
train.dist.pipeline_parallel_size = 4
train.dist.pipeline_num_layers = 24
train.dist.custom_pipeline_stage_id = [0]*6 + [1]*7 + [2]*7 + [3]*4
```
It means you have `[6, 7, 7, 4]` layers separately located in `stage0`~`stage3`.
Modify `custom_pipeline_stage_id` according to your own needs.

## Update Distributed Config with Command Line
You can also control the parallelization strategy by **command line** parameters as follows:

```bash
bash tools/train.sh tools/train_net.py configs/bert_large_pretrain.py \
8 \  # num of gpus
train.dist.data_parallel_size=2 \
train.dist.tensor_parallel_size=2 \
train.dist.pipeline_parallel_size=2 \
```