<img src="./assets/palm.gif" width="450px"></img>

# Pathways Language Model (PaLM) based on LiBai

Implementation of the model architecture of [Pathways Language Model (PaLM): Scaling to 540 Billion Parameters for Breakthrough Performance](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html) with OneFlow in less than <a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/PaLM/palm_model.py"> 300 lines of a code</a>.

We take advantage of [LiBai](https://github.com/Oneflow-Inc/libai) to exploit multiple parallelism strategies, e.g. data parallelism, tensor parallelism and pipeline parallelism. Besides, some advanced features such as mixed precision and ZeRO come here to help scale the training to multiple GPUs.

We welcome contributions, to help us enhance the usability of this project.

## Install

1. To install LiBai please follow <a href="https://libai.readthedocs.io/en/latest/tutorials/get_started/Installation.html">install instruction</a>.

2. Download the gpt dataset as demo dataset.

```shell
cd projects/PaLM
python tools/download_demo_dataset.py -o </PATH/TO/DATA>
```

## Usage

1. Configure your settings in `CONFIG_FILE.py` like below. We also provide some examples in [./configs](./configs/). You can check the <a hef="https://libai.readthedocs.io/en/latest/tutorials/basics/Features.html">advanced features tutorial</a> and <a hef="https://libai.readthedocs.io/en/latest/tutorials/basics/Distributed_Configuration.html">distributed tutorial</a>.

```python
train.amp.enabled = True
train.activation_checkpoint.enabled = True
```

2. Run

```shell
# single gpu
bash tools/train.sh tools/train_net.py projects/PaLM/configs/palm_pretrain.py 1

# 8 gpus
bash tools/train.sh tools/train_net.py projects/PaLM/configs/palm_pretrain.py 8
```
