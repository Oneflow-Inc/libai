## OneCls: OneFlow image classification toolbox based on LiBai
In this project, we try to combine [flowvision](https://github.com/Oneflow-Inc/vision) and [LiBai](https://github.com/Oneflow-Inc/libai) together and the users can easily write their own models in a torch-like way and enjoy the main features of [LiBai](https://github.com/Oneflow-Inc/libai), e.g., `nn.Graph`, `Data Parallel Training`, `ZeRO`, `FP16 Training` and so on.

## Usage
### Installation
Please see [LiBai Installation](https://libai.readthedocs.io/en/latest/tutorials/get_started/Installation.html) to install LiBai.

### Prepare the Data
Please see [Prepare the Data](https://libai.readthedocs.io/en/latest/tutorials/get_started/quick_run.html#prepare-the-data).

### Training
Here we use [DeiT](https://arxiv.org/abs/2012.12877) as an example, to train [DeiT](https://arxiv.org/abs/2012.12877) model on 8 GPUs:
```bash
cd /path/to/libai
bash tools/train.sh projects/OneCls/train_net.py projects/OneCls/configs/deit.py 8
```

### Train on different models registered in flowvision
You can easily change the training models by updating the [config](./configs/deit.py) file as follows:
```python
# config.py
...
# Change the model_name to train different models
model = LazyCall(VisionModel)(
    model_name = "swin_tiny_patch4_window7_224",
    pretrained = False,
    num_classes = 1000,
    loss_func = LazyCall(SoftTargetCrossEntropy)()
)
...
```
To train different models which are registered in flowvision, the users only need to update the `model.model_name` args. For more details about `Training & Evaluation`, please see [Training & Evaluation in Command Line](https://libai.readthedocs.io/en/latest/tutorials/basics/Train_and_Eval_Command_Line.html) for more details.

### Register your own model for training
Please see [REGISTER_MODELS.md](./REGISTER_MODELS.md) which teach the users how to register their own models for training.

## Model Zoo
Here is the supported model in flowvision, which can be directly imported in [config file](./configs/deit.py) for training.
- [x] [AlexNet](https://arxiv.org/abs/1404.5997)
- [x] [VGG](https://arxiv.org/abs/1409.1556)
- [x] [GoogleNet](https://arxiv.org/abs/1409.4842)
- [x] [ResNet](https://arxiv.org/abs/1512.03385)
- [x] [InceptionV3](https://arxiv.org/abs/1512.00567)
- [x] [SqueezeNet](https://arxiv.org/abs/1602.07360)
- [x] [DenseNet](https://arxiv.org/abs/1608.06993)
- [x] [ResNeXt](https://arxiv.org/abs/1611.05431)
- [x] [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [x] [MobileNetV3](https://arxiv.org/abs/1905.02244)
- [x] [ShuffleNetV2](https://arxiv.org/abs/1807.11164)
- [x] [SENet](https://arxiv.org/abs/1709.01507)
- [x] [Res2Net](https://arxiv.org/abs/1904.01169)
- [x] [MNASNet](https://arxiv.org/abs/1807.11626)
- [x] [EfficientNet](https://arxiv.org/abs/1905.11946)
- [x] [ResNeSt](https://arxiv.org/abs/2004.08955)
- [x] [GhostNet](https://arxiv.org/abs/1911.11907)
- [x] [RegNet](https://arxiv.org/abs/2003.13678)
- [x] [ViT](https://arxiv.org/abs/2010.11929)
- [x] [DeiT](https://arxiv.org/abs/2012.12877)
- [x] [PVT](https://arxiv.org/abs/2102.12122)
- [x] [Swin-Transformer](https://arxiv.org/abs/2103.14030)
- [x] [CSwin-Transformer](https://arxiv.org/abs/2107.00652)
- [x] [CrossFormer](https://arxiv.org/abs/2108.00154)
- [x] [PoolFormer](https://arxiv.org/abs/2111.11418)
- [x] [ResMLP](https://arxiv.org/abs/2105.03404)
- [x] [Mlp-Mixer](https://arxiv.org/abs/2105.01601)
- [x] [gMLP](https://arxiv.org/abs/2105.08050)
- [x] [ConvMixer](https://openreview.net/pdf?id=TVHS5Y4dNvM)
- [x] [ConvNeXt](https://arxiv.org/abs/2201.03545)
- [x] [RegionViT](https://arxiv.org/abs/2106.02689)
- [x] [VAN](https://arxiv.org/abs/2202.09741)
- [x] [LeViT](https://arxiv.org/abs/2104.01136)