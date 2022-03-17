# LiBai Model Zoo
To data, LiBai implements the following models:
- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [BERT](https://arxiv.org/abs/1810.04805)
- [T5](https://arxiv.org/abs/1910.10683)
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)


## Parallelism Mode in LiBai
In LiBai, a collection of parallel training strategies is provided:
- **Data Parallel Training**
- **Tensor Parallel Training**
- **Pipeline Parallel Training**

You can read oneflow official [tutorial](https://docs.oneflow.org/en/master/parallelism/01_introduction.html) to understand the basic conception about parallelization techniques.


## Supported Model in LiBai
In LiBai, you can try out different parallel modes easily by updating the [training config file](https://github.com/Oneflow-Inc/libai/blob/main/configs/common/train.py).
```python
dist=dict(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
)
```
For example, you can set `data_parallel_size=2` to split the input data into two groups for data parallel training.

For more details about the supported parallelism training on different models, please refer the following tables:

<table class="docutils">
  <tbody>
    <tr>
      <th width="80"> Model </th>
      <th valign="bottom" align="left" width="120">Data Parallel</th>
      <th valign="bottom" align="left" width="120">Tensor Parallel</th>
      <th valign="bottom" align="left" width="120">Pipeline Parallel</th>
    </tr>
    <tr>
      <td align="left"> <b> Vision Transformer </b> </td>
      <td align="left">&#10004;</td>
      <td align="left">&#10004;</td>
      <td align="left">&#10004;</td>
    </tr>
    <tr>
      <td align="left"> <b> Swin Transformer </b> </td>
      <td align="left">&#10004;</td>
      <td align="left">-</td>
      <td align="left">-</td>
    <tr>
      <td align="left"> <b> BERT </b> </td>
      <td align="left">&#10004;</td>
      <td align="left">&#10004;</td>
      <td align="left">&#10004;</td>
    </tr>
    <tr>
      <td align="left"> <b> T5 </b> </td>
      <td align="left">&#10004;</td>
      <td align="left">&#10004;</td>
      <td align="left">&#10004;</td>
    </tr>
    <tr>
      <td align="left"> <b> GPT-2 </b> </td>
      <td align="left">&#10004;</td>
      <td align="left">&#10004;</td>
      <td align="left">&#10004;</td>
    </tr>
    </tr>
  </tbody>
</table>

**Additions:**
&#10004; means you can train this model under specific parallelism techniques or combine two or three of them with &#10004; for 2D or 3D paralleism training.

**Examples:**
On above table, **BERT** model supports three parallelism techniques, if we have 1 node with 8 GPUs, you can try out different combinations of parallelism training techniques by updating [bert config file](../../../configs/bert_large_pretrain.py)` as follows:
- **Pure Data Parallel Training on 8GPUs**
```python
from .common.train import train
...

train.dist.data_parallel_size = 8
```
- **Pure Tensor Parallel Training on 8 GPUs**
```python
from .common.train import train
...

train.dist.tensor_parallel_size = 8
```
- **Pure Pipeline Parallel Training on 8 GPUs**
```python
from .common.train import train
...

train.dist.pipeline_parallel_size = 8
```
- **Data Parallel + Tensor Parallel for 2D Parallel Training on 8 GPUs**
```python
from .common.train import train
...

train.dist.data_parallel_size = 2
train.dist.tensor_parallel_size = 4
```
- **Data Parallel + Pipeline Parallel for 2D Parallel Training on 8 GPUs**
```python
from .common.train import train
...

train.dist.data_parallel_size = 2
train.dist.pipeline_parallel_size = 4
```
- **Tensor Parallel + Pipeline Parallel for 2D Parallel Training on 8 GPUs**
```python
from .common.train import train
...

train.dist.tensor_parallel_size = 2
train.dist.pipeline_parallel_size = 4
```
- **Data Parallel + Tensor Parallel + Pipeline Parallel for 3D Parallel Training on 8 GPUs**
```python
from .common.train import train
...

train.dist.data_parallel_size = 2
train.dist.tensor_parallel_size = 2
train.dist.pipeline_parallel_size = 2
```

You can also use command line to control the