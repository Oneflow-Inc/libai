# Advanced Tutorial

## Define your own parallel model with LiBai.layers

### Large-scale FC

Let's say that you have a huge fully-connected-layer for large-scale classification (e.g., 1000w classes), which makes it impossible to fit into a single GPU.

Don't worry, with help of `LiBai.layers`, you can write models in a familiar way that you used to write models for a single GPU. We give a simple example showing how to write a tensor-parallel fully-connected-layer with 2 GPUs.

```python
# huge_fc_example.py
import oneflow as flow
from omegaconf import DictConfig
from oneflow import nn

from libai.layers import Linear
from libai.utils import distributed as dist

cfg = DictConfig(dict(data_parallel_size=1, tensor_parallel_size=2, pipeline_parallel_size=1))
dist.setup_dist_util(cfg)


class Huge_FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(2048, 32768, parallel="col")

    def forward(self, x):
        return self.fc(x)


huge_fc = Huge_FC()

x = flow.rand(32, 2048, sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0, 1]))
y = huge_fc(x)

print(f"rank: {flow.env.get_rank()}, tensor shape: {y.to_local().shape}")
```

You can run this toy example with command line as follows:

```shell
python3 -m oneflow.distributed.launch --nproc_per_node 2 huge_fc_example.py

>> rank: 0, tensor shape: oneflow.Size([32, 16384])
>> rank: 1, tensor shape: oneflow.Size([32, 16384])
```

Through the result, you can find that `y` has been split along with `axis=1` on 2 GPUs.

### Large MLP models

MLP is very popular in transformer-based models. Assume we have a huge MLP model and its very large hidden size makes it difficult to fit into a single GPU.

We can then split the model weights across GPUs in a 2D mesh while you still write your model in a familiar way.

We give a simple example about the 2D parallel MLP in the LiBai context.

```python
import oneflow as flow
from omegaconf import DictConfig
from oneflow import nn

from libai.layers import Linear
from libai.utils import distributed as dist

cfg = DictConfig(dict(data_parallel_size=2, tensor_parallel_size=2, pipeline_parallel_size=1))
dist.setup_dist_util(cfg)

# Write a Simple 2D Parallel MLP
class MLP_2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = Linear(in_features=1024, out_features=16384, parallel="col")
        self.relu = nn.GELU()
        self.linear_2 = Linear(in_features=16384, out_features=1024, parallel="row")

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x

# define a model
mlp = MLP_2D()

# define input with 2D sbp
x = flow.rand(
    32,
    1024,
    sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
    placement=dist.get_layer_placement(0)
)
y = mlp(x)

print(f"rank: {flow.env.get_rank()}, tensor shape: {y.to_local().shape}")
```

You can run it with command line as follows:

```shell
python3 -m oneflow.distributed.launch --nproc_per_node 4 huge_mlp_example.py

>> rank: 2, tensor shape: oneflow.Size([16, 1024])
>> rank: 3, tensor shape: oneflow.Size([16, 1024])
>> rank: 1, tensor shape: oneflow.Size([16, 1024])
>> rank: 0, tensor shape: oneflow.Size([16, 1024])
```

From above, you can see that data are split into 2 groups for data parallel and weights are split into 2 groups for model parallel. So this simple example just implements a 2D parallel.

For the sake of your convenience, we provide some prevalent models such as BERT, GPT-2, and ViT in Mode Zoo. Feel free to customize them into different sizes to fit into your special needs.

## Pipeline Parallel

