# Features in LiBai

LiBai provides many features out of the box. We will show you how easily configuring them step by step.

## Automatic Mixed Precision Training

AMP stands for automatic mixed precision training. In LiBai, we allowed the user to train with AMP with just configuration. You can just simply add `amp` in your configuration file to use AMP.

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

Gradient clipping is a technique that tackles exploding gradients. The idea of gradient clipping is very simple: If the gradient gets too large, we rescale it to keep it small.

You do not have to worry about implementing gradient clipping when using LiBai, we support gradient clipping in a convenient way. You just need to add it in your configuration file.

> NOTE: We do not recommend users to write gradient clipping by themselves, because the naive gradient clipping may fail when using tensor parallel or pipeline parallel.

### Usage

```python
# import config
from .common.optim import optim

# get config
from libai.config import get_config
optim = get_config("common/optim").optim

# enable gradient clipping
optim.params.clip_grad_max_norm = 1.0
optim.params.clip_grad_norm_type = 2.0

# disable gradient clipping
optim.params.clip_grad_max_norm = None
optim.params.clip_grad_norm_type = None
```

`clip_grad_max_norm` and `clip_grad_norm_type` can be checked in [API docs of oneflow](https://oneflow.readthedocs.io/en/master/nn.html#oneflow.nn.utils.clip_grad_norm_)

## Gradient Accumulation

Gradient accumulation is a common strategy to enlarge your batch size for training. When training with large-scale models, memory can easily become the bottleneck. But decreasing the batch size (e.g., 2) will lead to unsatisfactory convergence.

Besides, when training with pipeline parallel, gradient accumulation make different stages executed in parallel micro-batch. Therefore, the calculation of each stage is not blocked.

### Usage

```python
# import config
from .common.train import train

# get config 
from libai.config import get_config
train = get_config("common/train").train

# enable grad accumulation for 4 steps
train.num_accumulation_steps = 4

# disable grad accumulation
train.num_accumulation_steps = None
```

## Activation Checkpointing

To reduce GPU memory usage and deploy a large model to a training system, LiBai support activation checkpointing. We use a Transformer layer as the unit of checkpointing because the activation size bloats in the middle of a Transformer layer so checkpointing the input of a Transformer layer is storage-efficient.

LiBai supported activation checkpointing by `set_activation_checkpoint` in `GraphBase`. So models using `libai.layers.TransformerLayer` supported activation checkpointing by default. If you want to set activation checkpointing for your customized layers, you need to override this function. 

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
train = get_config("common/train").train

# enable activation checkpointing
train.activation_checkpoint.enabled = True

# disable activation checkpointing
train.activation_checkpoint.enabled = False
```

## ZeRO 

Unlike basic data parallelism where memory states are replicated across data-parallel processes, Zero Redundancy Optimizer (ZeRO) partitions model states and gradients to save significant memory.

- Level 1: The optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition.

- Level 2: The reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states.

### Usage 

```python
# import config
from .common.train import train

# get config 
from libai.config import get_config
train = get_config("common/train").train

# enable zero 
train.zero_optimization.enabled = True

# enable zero for level-1
train.zero_optimization.stage = 1

# enable zero for level-2
train.zero_optimization.stage = 2

# disable zero
train.zero_optimization.enabled = False
```