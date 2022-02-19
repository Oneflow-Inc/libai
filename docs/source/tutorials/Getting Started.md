# Getting Started
This page guides you through a basic way to get started with LiBai:
- [Train VisionTransformer on ImageNet dataset](#train-visiontransformer-on-imagenet-dataset)
  - [Prepare the Data](#data-preparation)
  - [Train vit model from scratch](#train-vit-model-from-scratch)


## Train VisionTransformer on ImageNet dataset
### Prepare the Data
For ImageNet, we use standard ImageNet dataset, you can download it from http://image-net.org/.
- For standard folder dataset, move validation images to labeled sub-folders. The file structure should be like:
```bash
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...

```
### Train ViT model from scratch
- Update the data path in [vit_imagenet](https://github.com/Oneflow-Inc/libai/blob/main/configs/vit_imagenet.py) config file
```python
# Refine data path to imagenet data folder
dataloader.train.dataset[0].root = "/path/to/imagenet"
dataloader.test[0].dataset.root = "/path/to/imagenet"
```
- To train `vit_tiny_patch16_224` model on ImageNet on a single node with 8 GPUs for 300 epochs, run:
```bash
bash tools/train.sh configs/vit_imagenet.py 8
```
- The default vit model in LiBai is set to `vit_tiny_patch16_224`. To train other vit model you can update the [vit_imagenet](https://github.com/Oneflow-Inc/libai/blob/main/configs/vit_imagenet.py) config file by importing the other vit models in the config file as follows:
```python
# from .common.models.vit.vit_tiny_patch16_224 import model
from .common.models.vit.vit_base_patch16_224 import model
```
