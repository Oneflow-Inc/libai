# NeRF

Reproduce NeRF with OneFlow, which effect are equivalent to nerf_pytorch and nerf_pl [nerf_pytorch](https://github.com/yenchenlin/nerf-pytorch) and [nerf_pl](https://github.com/kwea123/nerf_pl).

## Introduce
The NeRF is used for 3D view rendering and [NeRF](https://arxiv.org/abs/2003.08934).

## Training NeRF
Training NeRF only on 1 GPUs.


### 1. Prepare the training data (blender and llff)
Prepare the training data by running:
Download `nerf_synthetic.zip` and `nerf_llff_data.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and unzip them.

NOTE: If your operating system without GUI is not convenient for pulling datasets, then please download the following script according
to (This script only downloads the blender dataset, to download the other dataset llff please change the filename and fileid):
```bash
#!/bin/bash
# Download zip dataset from Google Drive
filename='nerf_synthetic.zip'
fileid='18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

# Unzip
unzip -q ${filename}
rm ${filename}
```
### 2. Modify the dataset path and related hyperparameters in `projects/NeRF/configs/config_nerf.py`

### 3. Install additional dependencies
```bash
pip install opencv-python
pip install imageio
```
### 4. Run the following code to start training
```bash
# cd /path/to/libai
 bash tools/train.sh tools/train_net.py projects/NeRF/configs/config_nerf.py 1
```
Note that we use PSNR to evaluate the performance of NeRF. PSNR (Peak Signal-to-Noise Ratio) is an engineering term that represents the ratio of the maximum possible power of a signal to the destructive noise power that affects its representation accuracy.

### 5. Visual rendering results (Please modify the value of `train.load_weight` in `projects/NeRF/configs/config_nerf_for_rendering.py` first)
```bash
# cd /path/to/libai
 bash tools/train.sh tools/train_net.py projects/NeRF/configs/config_nerf_for_rendering.py 1 --eval-only
```
