Please refer to the [official repository](https://github.com/Oneflow-Inc/libai) and the [official documentation page](https://libai.readthedocs.io/en/latest/) for guidance on installation and other related topics.

## Running experiments in the OCCL paper
```shell
bash tools/train.sh tools/train_net.py configs/vit_imagenet.py <NUM_LOCAL_GPUS>
```

Notes:
- Prepare the ImageNet dataset in advance.
- Edit the [configs/vit_imagenet.py](configs/vit_imagenet.py#L84-L86) to switch among different distributed DNN training methods, following the guidelines in the [official doc](https://libai.readthedocs.io/en/latest/tutorials/basics/Distributed_Configuration.html).
- For training across multiple machines, edit the `NODE`, `NODE_RANK`, `ADDR`, and `ADDR_RANK` variables in [tools/train.sh](tools/train.sh#L8-L11).
- Edit [configs/vit_imagenet.py](configs/vit_imagenet.py#L2) to choose between the base ViT configuration or the large ViT configuration.
