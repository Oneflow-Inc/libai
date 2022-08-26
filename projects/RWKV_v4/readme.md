## AI-Writer RWKV-v4 in LiBai

This is the OneFlow re-implementation of [RWKV-v4](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v4) based on [LiBai](https://libai.readthedocs.io/).

## Supported parallel mode
Based on [libai.layers](https://libai.readthedocs.io/en/latest/modules/libai.layers.html), RWKV-v4 model is automatically configured with the following parallelism mode.

- Data Parallel
- Model Tensor Parallel

## Usage
### Installation
Please see [LiBai Installation](https://libai.readthedocs.io/en/latest/tutorials/get_started/Installation.html) to install LiBai

If torch pretrain model is needed, you need to install [pytorch](https://pytorch.org/)

### Prepare the Data
Download enwik8
```
cd your_data_dir
wget https://data.deepai.org/enwik8.zip
unzip enwik8.zip
```

### Torch pretrain model
We highly recommend you to use a torch pretrain model for initializing. you can save a `model.pth` from `https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v4`

### Pretraining
set your data path and your pretrain model path in [`configs/config.py`](./configs/config.py)

> your can set your model config to match pytorch model 

```python
model = LazyCall(GPT)(vocab_size=6064, ctx_len=1024, model_type="RWKV", n_layer=6, n_embd=512)

load_torch_checkpoint.path = "path/to/pytorch_model.pth"
datafile = "path/to/data/enwik8"
```

training RWKV_v4, refer to [LiBai training command doc](https://libai.readthedocs.io/en/latest/tutorials/basics/Train_and_Eval_Command_Line.html) for more details 
```shell
bash tools/train.sh projects/RWKV_v4/train_net.py projects/RWKV_v4/configs/config.py 1
```