## MOCOv3 in LiBai
**An Empirical Study of Training Self-Supervised Vision Transformers**

Xinlei Chen, Saining Xie, Kaiming He

[[`arXiv`](https://arxiv.org/abs/2104.02057)] [[`BibTeX`](#Citation)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/34954782/161363870-eb672518-deee-4754-b30f-be59ea91ac7e.png" width="480">
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
      <td align="left"> <b> MOCOv3 pretrain </b> </td>
      <td align="left">&#10004;</td>
      <td align="left">-</td>
      <td align="left">-</td>
    </tr>
    <tr>
      <td align="left"> <b> MOCOv3 linear prob </b> </td>
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
bash tools/train.sh projects/MOCOV3/pretrain_net.py projects/MOCOV3/configs/moco_pretrain.py 8
```

### Linear Prob
1. Setup the weights for linear prob in [moco_linear_prob.py](./configs/moco_linear_prob.py) as follows:

```python
# moco_linear_prob.py
# Path to the weight for linear prob
model.linear_prob = "path/to/pretrained_weight"
model.weight_style = "oneflow"
```
If you feel confused about the checkpoint format here, please refer to [Load and Save a Checkpoint in LiBai](https://libai.readthedocs.io/en/latest/tutorials/basics/Load_and_Save_Checkpoint.html) for more details.

2. The MOCOv3 linear prob on 8 GPUs using data parallelism.
```bash
cd /path/to/libai
bash tools/train.sh tools/train_net.py projects/MOCOV3/configs/moco_linear_prob.py 8
```
**Notes:** if you want to run the MOCOv3 linear prob models using different parallel strategies, please refer to the [Distributed Configuration Tutorial](https://libai.readthedocs.io/en/latest/tutorials/basics/Distributed_Configuration.html)


### Evaluation
Evaluate MOCOv3 model under LiBai on 8 GPUs:
```bash
cd /path/to/libai
bash tools/train.sh tools/train_net.py projects/MOCOV3/configs/moco_linear_prob.py 8 --eval-only train.load_weight="path/to/pretrained_weight"
```


## Advanced Usage
### The MOCOv3 linear prob with pytorch pretrained checkpoint
You can download pytorch pretrained weight from [MOCOv3 official repo](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md) and run linear prob in LiBai by updating the [moco_linear_prob.py](./configs/moco_linear_prob.py) as follows:
```python
# Path to the weight for linear prob 
model.linear_prob =  "/path/to/vit-s-300ep.pth.tar"
model.weight_style = "pytorch"
```
Run linear prob on 8 GPUs:
```bash
cd /path/to/libai
bash tools/train.sh tools/train_net.py projects/MOCOV3/configs/moco_linear_prob.py 8
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