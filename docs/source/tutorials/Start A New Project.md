# Start A New Project

If developers want to start a new project(Such as paper reproduction and finetune task), how can we use Libai to achieve the least code work.

The advantages of using Libai to develop a task are as follows:
- Avoid repeating a lot of work, many functions of Libai can be directly inherited in projects.
- We can quickly reproduce the experiments we have run, because Libai automatically saves the configuration file.
- Various information will be output during training, such as remaining training time, current iter, throughput, loss information and current learning rate.
- Only a few parameters need to be set for distributed training.


## Introduce
Let's use Bert_Finetune task as an example to introduce Libai, The complete project is [here](https://github.com/Oneflow-Inc/libai/tree/main/projects/QQP).

For start a new Libai project, our main work is as follows:

1. Prepare the file named `config.py`, relevant info as follows, you can refer to the [config.py](https://github.com/Oneflow-Inc/libai/blob/main/projects/QQP/configs/config_qqp.py).
    - This file is independent of the default config in Libai.
    - Contains the definition of task related Class, such as `Tokenizer` and `Dataset`.
    - Contains the relevant parameters of the task.
    - Defining related class with `LazyCall`, this method does not call the object, but returns a dict.
    - Many supporting objects have been imported from the `configs.common`, it can greatly reduce the workload.

2. Prepare the file named `model.py`, relevant info as follows, you can refer to the [model.py](https://github.com/Oneflow-Inc/libai/blob/main/projects/QQP/modeling/model.py).
    - Build related models in this file, the construction method is similar to OneFlow.
    - Because Libai will set up a static diagram by default, the calculation of loss needs to be inside the model.
    - The function `forward` must return a dict.
    - When defining a tensor in the model, need to use `to_global`, turn tensor into a consistent pattern.
    - When defining layers, you can import them directly from `libai.layers`, because it already have SBP defined.

3. Prepare the file named `dataset.py`, relevant info as follows, you can refer to the [dataset.py](https://github.com/Oneflow-Inc/libai/tree/main/projects/QQP/dataset).
    - Build `Dataset` in this file, the construction method is similar to OneFlow.
    - The difference is that we need to use `DistTensorData` and `Instance`.
    - The shape of each batch must be consistent.
    - The return `key`  
    - In `__getitem__` function, the `key` returned by the method must be consistent with the parameter name of the 'forward' function in the 'model'.


## Main Function Entry
[tools.train_net.py](https://github.com/Oneflow-Inc/libai/blob/main/tools/train_net.py) is the default main function entry provided in Libai, so we only need to rewrite some functions based on this file.

Here is an example:

```python
import numpy as np
from scipy.stats import spearmanr
from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.evaluation import DatasetEvaluator
from libai.trainer import DefaultTrainer, default_setup
from libai.utils import distributed as dist
from libai.utils.checkpoint import Checkpointer


def spearman_target(pred, labels):
    # Calculate spearman
    return spearmanr(pred, labels).correlation


class MyEvaluator(DatasetEvaluator):
    def __init__(self, cfg):
        self.cfg = cfg
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        pred = outputs["pred"]
        labels = outputs["labels"]
        self._predictions.append({"pred": pred, "labels": labels})

    def evaluate(self):
        if not dist.is_main_process():
            return {}
        else:
            predictions = self._predictions
        pred_array = np.array([])
        label_array = np.array([])
        for prediction in predictions:
            pred_array = np.append(pred_array, dist.tton(prediction["pred"]))
            label_array = np.append(label_array, dist.tton(prediction["labels"]))
        self._results = spearman_target(pred_array, label_array)
        return {"spearman": self._results}


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg):
        return MyEvaluator(cfg)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    trainer = MyTrainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
```

## Build Model
Preparing a model file is simple, a `model.py` example:

```python
from oneflow import nn

class MyModel(nn.Module):
    def __init__(self, cfg):
        ...

    def forward(self, input):
        ...
        return {'loss': loss}
```

## Build Datasets
Preparing a dataset file is simple, a `datasets.py` example:

```python
class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ...
        return Instance(
        input_ids = DistTensorData(
          flow.tensor(data["input_ids"], dtype=flow.long)
          ),
        attention_mask = DistTensorData(
          flow.tensor(data["attention_mask"], dtype=flow.long)
          ),
        )
```

## Build Config
The `config.py` in Libai is special, which takes the form of lazyconfig and will be saved as `.yaml` at runtime.
The following describes the complete `config.py` and how to inherit the config in Libai.

First, config has several necessary fields:
- `train`: It contains training related parameters and is a dict type.
- `model`: Model used by the task, specify the generation method in the file, due to the characteristics of lazycall, the model     will be generated at runtime.
- `optim`: Optimizer related. If not defined, the default will be used.
- `lr_scheduler`: Related to learning rate, If not defined, the default will be used.
- `graph`: Import directly, and the model will be automatically converted to graph during operation.

> All imported modules must take Libai as the root directory, otherwise, the saved `yaml` file will not be able to save the correct path of the module, resulting in an error when reading `yaml`, so the experiment cannot be reproduced.

A complete `config.py` example:

```python
from omegaconf import OmegaConf
from configs.common.data.bert_dataset import tokenization
from configs.common.models.bert import cfg as my_cfg
from configs.common.models.graph import graph
from configs.common.optim import optim
from configs.common.train import train
from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from libai.tokenizer import BertTokenizer
from libai.optim import get_default_optimizer_params, PolynomialLR
from projects.MyProjects.dataset.dataset import TrainDataset, TestDataset
from projects.MyProjects.modeling import MyModel


tokenization.tokenizer = LazyCall(BertTokenizer)(
    vocab_file=".../vocab.txt",
)

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(TrainDataset)(
            path=".../train.txt",
            ...
            ),
        )
    ],
)

dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(TestDataset)(
            path=".../test.txt",
            ...
            ),
        ),
    ),
]

my_cfg.update(
    dict(
        vocab_size=21128,
        ...
    )
)

model = LazyCall(MyModel)(cfg=my_cfg)

optim = LazyCall(flow.optim.AdamW)(
    parameters=LazyCall(get_default_optimizer_params)(
        # parameters.model is meant to be set to the model object, before instantiating the optimizer.
        clip_grad_max_norm=1.0,
        ...
    ),
    lr=1e-4,
    ...
)

lr_scheduler = LazyCall(flow.optim.lr_scheduler.WarmUpLR)(
    lrsch_or_optimizer=LazyCall(PolynomialLR)(steps=1000, end_learning_rate=1.0e-5,),
    warmup_factor=0,
    ...
)

train.update(
    dict(
        output_dir=".../result",
        train_micro_batch_size=64,
        ...
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
    )
)

```

After building the `config.py`, if we want to get the corresponding fields in the project, we just need to access like `cfg.my_cfg.***`

## Start Training
After the above modules are built, we can start training.

```bash
bash projects/my_projects/train.sh projects/my_projects/config.py 1
```

> Config can support both `py` files and generated `yaml` files.

```bash
CONFIG=projects/your_task/config.py  # output/your_task/config.yaml
GPUS=1
NODE=1
NODE_RANK=0
PORT=2345

python3 -m oneflow.distributed.launch \
    --nproc_per_node $GPUS \
    --nnodes $NODE \
    --node_rank $NODE_RANK \
    --master_addr $PORT \
    projects/your_task/finetune.py \
    --config-file $CONFIG \
    --num-gpus $GPUS
```