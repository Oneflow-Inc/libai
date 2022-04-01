# Build New Project on LiBai

This is a basic guide to build new projects based on LiBai. The advantages of using LiBai to start a new project (such as paper reproduction and finetune task) are as follows:

- Avoid redundant work. Developers can directly inherit many built-in modules from LiBai.
- Easily reproduce the experiments already run, because LiBai will save the configuration file automatically.
- Automatically output useful information during training time, such as remaining training time, current iter, throughput, loss information and current learning rate, etc.
- Set a few config params to enjoy distributed training techniques.

## Introduction
Take the [Bert Finetune](https://github.com/Oneflow-Inc/libai/tree/main/projects/QQP) task as an example to introduce LiBai.

The complete file structure of the project is:

```
projects/my_project
├── configs
│   └── config_custom.py
│   └── ...
├── dataset
│   ├── custom_dataset.py
│   └── ...
├── modeling
│   ├── custom_model.py
│   └── ...
├── README.md
```

To start a new project based on LiBai step by step:

Step 1. Prepare an independent config file (such as [config.py](https://github.com/Oneflow-Inc/libai/blob/main/projects/QQP/configs/config_qqp.py)) which contains:

- The relevant parameters of the task.
- The pre-defined related Class, such as `Model`, `Optimizer`, `Scheduler`, `Dataset`.
- You can inherit the default config in `configs/common` and rewrite it, which can greatly reduce the workload.
- Related class defined with LazyCall which returns a dict instead of calling the object.

Step 2. Prepare a model file (such as [model.py](https://github.com/Oneflow-Inc/libai/blob/main/projects/QQP/modeling/model.py)) :
- Build related models in this file. The construction method is similar to OneFlow.
- Because Libai will set up a static diagram by default, the calculation of loss needs to be inside the model.
- The function `forward` must return a dict.
- When defining a tensor in the model, you need to use `to_global`. Turn tensor into a global pattern.
- When defining layers, you can import them directly from `libai.layers`, because it has already pre-defined the SBP signature.

Step 3. Prepare a dataset file (such as [dataset.py](https://github.com/Oneflow-Inc/libai/tree/main/projects/QQP/dataset)) :
- Build `Dataset` in this file. The construction method is similar to OneFlow.
- The difference is that you need to use `DistTensorData` and `Instance`.
- The shape of each batch must be global.
- In `__getitem__` function, the `key` returned by the method must be consistent with the parameter name of the `forward` function in the `model`.


## Main Function Entry
[tools/train_net.py](https://github.com/Oneflow-Inc/libai/blob/main/tools/train_net.py) is the default main function entry provided in LiBai.


## Build Config
The `config.py` in LiBai is special, which takes the form of lazyconfig and will be saved as `.yaml` at runtime. The config has several necessary fields, such as `train`, `model`, `optim`, `lr_scheduler`, `graph`. For more information, please refer to [Config_System.md](https://libai.readthedocs.io/en/latest/tutorials/Config_System.html).

> All imported modules must take LiBai as the root directory. Otherwise, the saved `yaml` file cannot save the correct path of the module, resulting in an error when reading `yaml`, and the experiment cannot be reproduced.

After building the `config.py`, if you want to get the corresponding fields in the project, you need to access like `cfg.my_cfg.***`.

## Start Training
The `train.sh` file contains some parameters, such as `GPUS`, `NODE`, etc.

```bash
#!/usr/bin/env bash
FILE=$1
CONFIG=$2
GPUS=$3
NODE=${NODE:-1}
NODE_RANK=${NODE_RANK:-0}
ADDR=${ADDR:-127.0.0.1}
PORT=${PORT:-12345}

python3 -m oneflow.distributed.launch \
--nproc_per_node $GPUS --nnodes $NODE --node_rank $NODE_RANK --master_addr $ADDR --master_port $PORT \
$FILE --config-file $CONFIG ${@:4}
```

After building the above modules, you can start training with single GPU.

> Config can support both `py` files and generated `yaml` files.

```bash
bash projects/my_projects/train.sh tools/train_net.py projects/my_projects/config.py 1
```
