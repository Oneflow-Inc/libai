# Features in LiBai

LiBai provides many features out of the box. This section shows how to configure them step by step.

## Automatic Mixed Precision Training

AMP stands for automatic mixed precision training. To enable AMP in LiBaiYou, add `train.amp.enabled=True` in your configuration file .

### Usage

```python
# import config
from .common.train import train

# get config
from libai.config import get_config
train = get_config("common/train.py").train

# enable amp
train.amp.enabled = True
# disable amp
train.amp.enabled = False
```

## Gradient Clipping

Gradient clipping is a technique that tackles exploding gradients. The idea of gradient clipping is very simple: the gradient will be rescaled down if it gets too large.

LiBai supports gradient clipping in a convenient way, and you don't have to implement it by yourself. You just need to add some settings to your configuration file to enable gradient clipping.

**Note:** We do not recommend writing gradient clipping by yourself, because naive gradient clipping may fail when using tensor parallel or pipeline parallel.

### Usage

```python
# import config
from .common.optim import optim

# get config
from libai.config import get_config
optim = get_config("common/optim.py").optim

# enable gradient clipping
optim.params.clip_grad_max_norm = 1.0
optim.params.clip_grad_norm_type = 2.0

# disable gradient clipping
optim.params.clip_grad_max_norm = None
optim.params.clip_grad_norm_type = None
```

`clip_grad_max_norm` and `clip_grad_norm_type` can be checked in [API docs of oneflow](https://oneflow.readthedocs.io/en/master/nn.html#oneflow.nn.utils.clip_grad_norm_).

## Gradient Accumulation

Gradient accumulation is a common strategy to train large-scale models when memory becomes the bottleneck. This technique splits the mini-batch into several micro-batches, then performs normal forward and backward operations. Models will only be updated after accumulating the gradients of all these micro-batches.

Besides, when training with pipeline parallel, gradient accumulation makes different stages executed in different micro-batch in parallel. Therefore, the calculation of each stage can be overlapped.

### Usage

```python
# import config
from .common.train import train

# get config 
from libai.config import get_config
train = get_config("common/train.py").train

# enable grad accumulation for 4 steps
train.num_accumulation_steps = 4

# disable grad accumulation
train.num_accumulation_steps = None
```

## Activation Checkpointing

To reduce GPU memory usage and deploy a large model to a training system, LiBai supports activation checkpointing. LiBai uses a Transformer layer as the unit of checkpointing, because the activation size bloats in the middle of a Transformer layer, so checkpointing the input of a Transformer layer is storage-efficient.

LiBai supports [activation checkpointing](https://arxiv.org/abs/1604.06174) by `set_activation_checkpoint` in `GraphBase`. So models using `libai.layers.TransformerLayer` support activation checkpointing by default. If you want to set activation checkpointing for customized layers, you need to override this function.

```python
def set_activation_checkpoint(self):
    for module_block in self.model.modules():
        if isinstance(module_block.origin, TransformerLayer):
            module_block.config.activation_checkpointing = True
```

### Usage

```python
# import config
from .common.train import train

# get config 
from libai.config import get_config
train = get_config("common/train.py").train

# enable activation checkpointing
train.activation_checkpoint.enabled = True

# disable activation checkpointing
train.activation_checkpoint.enabled = False
```

## ZeRO 

Unlike normal data parallelism, where model states and gradients are replicated across data-parallel processes, Zero Redundancy Optimizer (ZeRO) partitions them across data-parallel processes, which can reduce memory footprint significantly.

- Level 1: The optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first and second moment estimates) are partitioned across the processes so that each process will only update its own partition.

- Level 2: The reduced 32-bit gradients for updating the model weights are also partitioned so that each process retains only the gradients corresponding to its portion of the optimizer states.

> **Note:** ZeRO only supports data parallel and pipeline parallel, or the combination of them. If you use tensor parallel in your training, make sure ZeRO is disabled.

### Usage 

```python
# import config
from .common.train import train

# get config 
from libai.config import get_config
train = get_config("common/train.py").train

# enable zero 
train.zero_optimization.enabled = True

# enable zero for level-1
train.zero_optimization.stage = 1

# enable zero for level-2
train.zero_optimization.stage = 2

# disable zero
train.zero_optimization.enabled = False
```