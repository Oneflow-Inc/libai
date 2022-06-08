# Auto Parallel Training

LiBai supports **auto-parallel training** which means LiBai will automatically find **an efficient parallel training strategy** for a specific model during training. Users can try out auto-parallel training by the following steps.

## Installation
Install OneFlow Auto-Parallel Branch

```shell
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/release-auto_parallel-v0.1/[PLATFORM]
```
- All available `[PLATFORM]`:

<table class="docutils">
  <tbody>
    <tr>
      <th width="80"> Platform </th>
      <th valign="bottom" align="left" width="120">CUDA Driver Version</th>
      <th valign="bottom" align="left" width="120">Supported GPUs</th>
    </tr>
    <tr>
      <td align="left"> cu112 </td>
      <td align="left"> >= 450.80.02 </td>
      <td align="left"> GTX 10xx, RTX 20xx, A100, RTX 30xx</td>
    </tr>
    <tr>
      <td align="left"> cu102 </td>
      <td align="left"> >= 440.33 </td>
      <td align="left"> GTX 10xx, RTX 20xx</td>
    </tr>
    <tr>
      <td align="left"> cpu </td>
      <td align="left"> N/A </td>
      <td align="left"> N/A </td>
    </tr>
  </tbody>
</table>


## Train/Evaluate model in auto-parallel mode
You can train your own model in auto-prallel mode by simply updating the config as follows:
### Modify config file
```python
# your config
from .common.models.graph import graph

graph.auto_parallel.enabled = True
```
Training model with auto-parallel on 4 GPUs:
```shell
bash ./tools/train.sh tools/train_net.py configs/your_own_config.py 4
```

### Directly modify the training command line
- auto-parallel training:
```shell
bash ./tools/train.sh tools/train_net.py configs/your_own_config.py 4 graph.auto_parallel.enabled=True
```

- auto-parallel evaluation:

```shell
bash ./tools/train.sh tools/train_net.py configs/your_own_config.py 4 --eval graph.auto_parallel.enabled=True
```