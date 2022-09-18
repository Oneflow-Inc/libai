# NeRF

Reproduce NeRF with OneFlow, which effect are equivalent to nerf_pytorch and nerf_pl [nerf_pytorch](https://github.com/yenchenlin/nerf-pytorch) and [nerf_pl](https://github.com/kwea123/nerf_pl).

## Introduce
The NeRF is used for 3D view rendering and [NeRF](https://arxiv.org/abs/2003.08934).

## Training NeRF
Training NeRF only on 1 GPUs.


### 1. Prepare the training data (blender and llff)

Prepare the training data by running:

Download `nerf_synthetic.zip` and `nerf_llff_data.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and unzip them.

### 2. Run the following code to start training
```bash
# cd /path/to/libai
 bash tools/train.sh tools/train_net.py projects/NeRF/configs/config_nerf.py 1
```
### 3. Visual rendering results
```bash
# cd /path/to/libai
 bash tools/train.sh tools/train_net.py projects/NeRF/configs/config_nerf_for_rendering.py 1 --eval-only
```
