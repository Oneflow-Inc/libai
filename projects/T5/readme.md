# T5

Note: this project uses a special Dataset, which is an **unofficial** version, if you want to use your own dataset and experience complete mt5, please refer to [projects/MT5](../MT5/)

Reproduce T5Model and MT5Model with OneFlow, which effect is equivalent to HuggingFace's [T5](https://huggingface.co/docs/transformers/v4.19.4/en/model_doc/t5#overview) and [T5v1.1](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511).

## Introduce
The T5 pretraining project can support 3D parallel and [ZERO](https://arxiv.org/abs/2202.10435).

## Training MT5
Training MT5 on 8 GPUs using 3D parallelism and ZERO.

### 1. Prepare your training config file

set the pretrain parameters in `T5/configs/mt5_pretrain.py`, such as `train_data_path` and `pretrained_model_path`.
> If the `pretrained_model_path` is set, the path should contain `pytorch_model.bin` and `config.json`,
> and the `T5/configs/mt5_pretrain.py` set default `pretrained_model_path` for consistent the same initialization as huggingface.

### 2. Prepare the default init model and config

```bash
# /path/to/projects/T5/data/init_mt5/
wget http://oneflow-static.oss-cn-beijing.aliyuncs.com/libai/mt5_init/config.json
wget http://oneflow-static.oss-cn-beijing.aliyuncs.com/libai/mt5_init/pytorch_model.bin
```

### 3.Prepare the training data

```bash
# /path/to/projects/T5/data/training_data/
wget http://oneflow-static.oss-cn-beijing.aliyuncs.com/libai/mt5_init/wudao_180g_test_bert_tokenized_512_demo.zip
unzip wudao_180g_test_bert_tokenized_512_demo.zip
```

The structure of the training data folder should be like:
```
$ tree training_data_dir
path/to/projects/T5/data/training_data/
 ├──part_0
 ├──part_1
 ├──part_2
 ├──part_3
     ...
 ├──part_8
 └──part_9 
```

### 4. Run the following code
```bash
# cd /path/to/libai
bash tools/train.sh tools/train_net.py projects/T5/configs/mt5_pretrain.py 8
```

### 5. Convert OneFlow Checkpoint to PyTorch
Suppose you need to put the final checkpoint to huggingface, run the following code, and ensure the parameters are set correctly in `weight_convert.sh`:

```bash
# cd /path/to/libai
bash projects/T5/utils/weight_convert.sh
```