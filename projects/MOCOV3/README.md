## MOCOv3 in LiBai
**An Empirical Study of Training Self-Supervised Vision Transformers**

Xinlei Chen, Saining Xie, Kaiming He

[[`arXiv`](https://arxiv.org/abs/2104.02057)] [[`BibTeX`](#Citation)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>

This is the OneFlow re-implementation of MOCOv3 based on [LiBai](https://libai.readthedocs.io/).

## Catelog
- [x] MOCOv3 pretraining code
- [x] MOCOv3 linear prob code

## Supported parallel mode and task
Based on [libai.layers](https://libai.readthedocs.io/en/latest/modules/libai.layers.html), MOCOv3 model is automatically configured with the following parallelism mode.

<table class="docutils">
  <tbody>
    <tr>
      <th width="80"> Model </th>
      <th valign="bottom" align="left" width="120">Data Parallel</th>
      <th valign="bottom" align="left" width="120">Tensor Parallel</th>
      <th valign="bottom" align="left" width="120">Pipeline Parallel</th>
    </tr>
    <tr>
      <td align="left"> <b> MAE pretrain </b> </td>
      <td align="left">&#10004;</td>
      <td align="left">-</td>
      <td align="left">-</td>
    </tr>
    <tr>
      <td align="left"> <b> MAE finetune </b> </td>
      <td align="left">&#10004;</td>
      <td align="left">&#10004;</td>
      <td align="left">&#10004;</td>
    </tr>
  </tbody>
</table>


## Usage
### Installation
Please see [LiBai Installation](https://libai.readthedocs.io/en/latest/tutorials/get_started/Installation.html) to install LiBai

### Prepare the Data
Please see [Prepare the Data](https://libai.readthedocs.io/en/latest/tutorials/get_started/quick_run.html#prepare-the-data).


### Pretraining
Pretraining MOCOv3 on 8 GPUs using data parallelism.
```bash
cd /path/to/libai
bash tools/train.sh projects/MOCOV3/pretrain_net.py projects/MOCOV3/moco_pretraining.py 8
```

### Linear Prob
1. Setup the weights for finetuning in [moco_finetune.py](./configs/moco_finetune.py) as follows:

```python
# moco_funetune.py
finetune.enable = True  # only load weight if enable is True
finetune.weight_style = "oneflow"  # Set "oneflow" for loading oneflow checkpoints
finetune.path = "/path/to/checkpoint"  # the checkpoint directory
```
If you feel confused about the checkpoint format here, please refer to [Load and Save a Checkpoint in LiBai](https://libai.readthedocs.io/en/latest/tutorials/basics/Load_and_Save_Checkpoint.html) for more details.

2. Finetune MOCOv3 on 8 GPUs using data parallelism.
```bash
cd /path/to/libai
bash tools/train.sh projects/MOCOV3/finetune_net.py projects/MAE/moco_finetune.py 8
```
**Notes:** if you want to finetune MOCOv3 models using different parallel strategies, please refer to the [Distributed Configuration Tutorial](https://libai.readthedocs.io/en/latest/tutorials/basics/Distributed_Configuration.html)


### Evaluation
Evaluate MOCOv3 model under LiBai on 8 GPUs:
```bash
cd /path/to/libai
bash tools/train.sh projects/MOCOV3/finetune_net.py projects/MOCOV3/moco_finetune.py 8 --eval-only
```


## Advanced Usage
### Finetune MOCOv3 with pytorch pretrained checkpoint
You can download pytorch pretrained weight from [MOCOv3 official repo](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md) and finetune them in LiBai by updating the [moco_finetune.py](./configs/moco_finetune.py) as follows:
```python
finetune.enable = True  # only load weight if enable is True
finetune.weight_style = "pytorch"  # Set "pytorch" for loading torch checkpoints
finetune.path = "/path/to/vit-s-300ep.pth.tar"
```
Run finetuning on 8 GPUs:
```bash
cd /path/to/libai
bash tools/train.sh projects/MOCOV3/finetune_net.py projects/MOCOV3/moco_finetune.py 8
```


## Citation
```BibTeX
@inproceedings{chen2021empirical,
  title={An empirical study of training self-supervised vision transformers},
  author={Chen, Xinlei and Xie, Saining and He, Kaiming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9640--9649},
  year={2021}
}
```