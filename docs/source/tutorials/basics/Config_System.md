# Config System

Given that the traditional yacs-based config system or python argparse command-line options suffer from providing enough flexibility for the development of new project, we borrowed the [lazy config system](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) design from detectron2 which forms the non-intrusive config system for LiBai.

You can refer to the [d2 tutorial](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) for more details about the syntax and basic usage of lazy config. This section shows some examples of usage in LiBai.

## Configs in LiBai

LiBai defines a standard set of config namespaces for later use. This set of namespaces must be kept if you want to perform the complete training and evaluation process of LiBai. 

In summary, this set of namespaces is `model, graph, train, optim, dataloader, tokenization(optional)`. The details are as follows.

### model

This is the configuration for model definition. You can refer to `configs/common/models` for more examples.

A model config file can be loaded like this:

```python
# bert.py:
from libai.config import LazyCall
from libai.models import BertModel

# define a model with lazycall
bert_model = LazyCall(BertModel)(
    vocab_size=30522,
    hidden_size=768,
    hidden_layers=24,
    num_attention_heads=12,
    intermediate_size=4096,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    num_tokentypes=2,
    add_pooling_layer=True,
    initializer_range=0.02,
    layernorm_eps=1e-5,
    bias_gelu_fusion=True,
    bias_dropout_fusion=True,
    scale_mask_softmax_fusion=False,
    apply_query_key_layer_scaling=True,
    add_binary_head=True,
    amp_enabled=False,
)

# my_config.py:
from bert import bert_model as model
assert model.hidden_size == 768
model.hidden_layers = 12 # change hidden layers
```

After defining the model config in a python file, you can `import` it in the global scope of the config file. Note that you need to rename it as `model` regardless of the name used in the model config.

You can access and change all keys in the model config after import.

### graph

This is the configuration for static `nn.Graph` mode. For more information about the static graph mode, refer to the official [nn.Graph docs](https://docs.oneflow.org/master/basics/08_nn_graph.html).

LiBai has already defined a `GraphBase` class for almost all models to use. You can simply turn on this option to convert eager mode to graph mode.

The graph config can be found in [graph.py](https://github.com/Oneflow-Inc/libai/blob/main/configs/common/models/graph.py), and two useful options are shown as follows:

```python
# Turn on graph mode, if set to `False`, will use eager mode.
graph.enabled = True 

# Set graph debug level, -1 means no debug info, and 0,1,2,3 can be 
# set for different debug levels. 
# More information can be found in nn.Graph documents.
graph.debug = -1 
```

### train

This is the configuration for training and evaluation. The default training config can be found in `configs/common/train.py`.

The convention of training / test specific parameters is as follows:

```python
from libai.config import LazyCall

train = dict(
    
    # Directory where output files are written
    output_dir="./output",

    # `train_micro_batch_size` is number of samples per batch on each GPU. 
    # train_mini_batch_size = train_micro_batch_size * num_accumulation_steps.
    # This is also the number of training samples per step (i.e. per iteration). 

    # If we use 8 GPUs for data parallel groups, `train_micro_batch_size = 2` and 
    # `num_accumulation_steps = 4`, then each GPU will see 2 samples per batch and
    # 8 samples per iteration.
    # Total 64 samples will be trained per iteration across all GPUs.

    # global_batch_size = micro_batch_size  * num_grad_acc * data_parallel_groups
    train_micro_batch_size=32,
    global_batch_size=None,
    num_accumulation_steps=None,

    # The total training iterations
    train_iter=10000,
    # The total training epochs, will be scaled to training iterations automatically.
    # The actual total training iterations will be calculated by the 
    # formula `max(train_iter, train_epoch * iter_per_epoch)`.
    train_epoch=0,  
    consumed_train_samples=0,
    consumed_valid_samples=0,
    train_samples=None,

    # Fraction of lr-warmup-iters to use for warmup (as a float)
    warmup_ratio=0,  

    # The start iteration, usually needn't set it manually.
    # It can be computed automatically when resuming training.
    start_iter=0,

    # Enable automatic mixed precision for training which does not 
    # change model's inference behavior.
    amp=dict(enabled=False),  

    # Enable activation checkpointing to allow for training
    # with larger models, sequences, and batch sizes.
    # If enabled, checkpoint the input activations of each transformer layers by default.
    activation_checkpoint=dict(enabled=False),  

    # NCCL fusion threshold megabytes, set to 0 to 
    # compatible with previous version of OneFlow.
    nccl_fusion_threshold_mb=16,

    # Maximum number of ops of NCCL fusion, set to 0 to 
    # compatible with previous version of OneFlow.
    nccl_fusion_max_ops=24,

    # Enable ZeRO Optimization to allow for training with larger models.
    # This optimization will reduce optimizer stages memory consumption
    # as described in ZeRO https://arxiv.org/abs/1910.02054.
    zero_optimization=dict(
        enabled=False,
        stage=1,
    ),
    
    # Save a model checkpoint after every this number of iterations,
    # and maximum number of checkpoint will be kept.
    checkpointer=dict(period=5000, max_to_keep=100),  

    # Options for evaluation

    # `test_micro_batch_size` is number of samples per batch on each GPU for testing. 
    # If we use 8 GPUs for data parallel groups and `test_micro_batch_size = 2`, then
    # total 16 samples will be used per iteration across all GPUs.
    test_micro_batch_size=32,

    # Enabled evaluation during training, after every `eval_period` number of iterations
    # will perform the evaluation process.
    # You can set the maximum evaluation iterations to run for validation/test.
    # You can also set a customized evaluator for use.
    evaluation=dict(
        enabled=True,
        # evaluator for calculating top-k acc
        evaluator=LazyCall(ClsEvaluator)(topk=(1, 5)),  
        eval_period=5000,
        eval_iter=1e9,  # running steps for validation/test

        # Metrics to be used for best model checkpoint.
        eval_metric="Acc@1",
        eval_mode="max",
    ),

    # Path to a checkpoint file to be loaded to the model for training or evaluation. 
    load_weight="",

    # Output log to console after every this number of iterations.
    log_period=20,

    # lr_scheduler arguments
    # See libai/scheduler/lr_scheduler.py for definition.
    scheduler=LazyCall(WarmupCosineLR)(
        # In DefaultTrainer we will automatically set `max_iter`
        # and `warmup_iter` by the given train cfg.
        warmup_factor=0.001,
        alpha=0.01,
        warmup_method="linear",
    ),

    # Distributed arguments
    # See https://libai.readthedocs.io/en/latest/tutorials/basics/Distributed_Configuration.html for more details.
    dist=dict(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        # users must set the `pipeline_num_layers` attribute when `pipeline_parallel_size > 1`
        pipeline_num_layers=None,
        # users could customize the number of layers in different stages
        # by setting the `custom_pipeline_stage_id ` attribute which is used for
        # manually balance calculation between stages when running pipeline parallelism
        # e.g. you can set `custom_pipeline_stage_id=[0, 0, 0, 1]`
        # for `pipeline_num_layers=4 and pipeline_parallel_size=2`
        # which means the first 3 layers will be placed on stage0 and
        # the last layer will be placed on stage1
        # NOTE: if it is None, LiBai will automatically set pipeline_stage_id
        # `auto_pipeline_stage_id` and `actual_pipeline_stage_id` will be saved in `config.yaml`
        custom_pipeline_stage_id=None,
    ),

    # the device type of input tensors for model, defaults to "cuda".
    # if you want to accelerate the model training when pipeline_parallel > 1
    # you can set `input_placement_device="cpu"` then call input_tensor.to_global()
    # inside your model.forward() method
    # see `libai/models/bert_model.py` as reference
    input_placement_device="cuda",

    # set to `True` to enable rdma for improving speed of pipeline_parallel
    rdma_enabled=True,
    
    # Set seed to positive to use a fixed seed. Note that a fixed seed increases
    # reproducibility but does not guarantee fully deterministic behavior.
    # Disabling all parallelism further increases reproducibility.
    seed=1234,
)
```
**Note:** ``warmup_ratio`` is the ratio of warmup iterations of the total training iterations, and the real ``warmup iterations`` will be calculated by ``wramup_ratio * train_iter`` automatically.

**Example:** If you need to train 300 epochs with 5 warmup epochs, update the config as follows:
```python
# config.py
train.train_epoch = 300
train.warmup_ratio = 5 / 300
```
If you need to train 1000 iters with 200 warmup iters, set the training config like this:
```python
# config.py
train.train_iter = 1000
train.warmup_ratio = 200 / 1000
```


### optim

This is the configuration for optimizer. The default configuration can be found in `configs/common/optim.py`.

LiBai utilizes the function `get_default_optimizer_params`, which needs the `nn.Module` as the argument and returns the parameter groups.

With `LazyConfig`, you can set other arguments in advance and pass the `model` argument later. For more details, refer to [API docs of libai optim](../libai.optim.html#libai.optim.get_default_optimizer_params).

```python
# optim.py:
import oneflow as flow
from libai.config import LazyCall
from libai.optim import get_default_optimizer_params

optim = LazyCall(flow.optim.AdamW)(
    params=LazyCall(get_default_optimizer_params)(
        # params.model is meant to be set to the model object,
        # before instantiating the optimizer.
        clip_grad_max_norm=1.0,
        clip_grad_norm_type=2.0,
        weight_decay_norm=0.0,
        weight_decay_bias=0.0,
    ),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
    do_bias_correction=True,
)

# my_config.py:
import oneflow as flow
optim._target_ = flow.optim.SGD

# Remove the incompatible arguments in optim
del optim.do_bias_correction

# Set the need arguments
optim.momentum = 0.9
```

### dataloader

This is the configuration for dataset/dataloader. This component provides data to the model. A dataloader usually takes raw information and processes it into the format required by the model.

See example datasets in `configs/common/data/`, including `cifar100`, `imagenet`, `bert_dataset` and so on. You can also define your customized dataset config as you like.

Take `bert_dataset.py` as an example:

```python
# bert_dataset.py:
from libai.config import LazyCall
from omegaconf import OmegaConf
from libai.data import build_nlp_test_loader, build_nlp_train_val_test_loader
from libai.data.datasets import BertDataset
from libai.data.data_utils import get_indexed_dataset

dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_nlp_train_val_test_loader)(
    dataset=[
        LazyCall(BertDataset)(
            data_prefix="/your/data_prefix/path",
            indexed_dataset=LazyCall(get_indexed_dataset)(
                data_prefix="/your/data_prefix/path",
                data_impl="mmap",
                skip_warmup=False,
            ),
            max_seq_length=512,
            mask_lm_prob=0.15,
            short_seq_prob=0.1,
        ),
    ],
    splits=[[949.0, 50.0, 1.0]],
    weights=[1.0],
    num_workers=4,
)

# my_config.py:
dataloader.train.dataset[0].max_seq_length = 256
dataloader.train.num_workers = 2
```

LiBai provides two functions `build_nlp_train_val_test_loader` and `build_image_train_loader` to create a default train data loader from a given config. It takes the list of `dataset_class`(e.g., `BertDataset`) and combines them using `flow.utils.data.dataset.ConcatDataset`. 

It is recommended to check out [API docs of libai.data](../libai.data.html#libai.data.build.build_nlp_train_loader) to learn more about the APIs of `build_nlp_train_val_test_loader`.

### tokenization (optional)

You need to configure a tokenizer if you want to train a NLP task. Each NLP dataset has its own tokenizer config in the corresponding data config file.

Here we use:

```python
# bert_dataset.py:
from libai.config import LazyCall
from omegaconf import OmegaConf
from libai.tokenizer import BertTokenizer

tokenization = OmegaConf.create()

tokenization.tokenizer = LazyCall(BertTokenizer)(
    vocab_file="bert-base-chinese-vocab.txt",
    do_lower_case=True,
    do_chinese_wwm=True,
)
tokenization.append_eod = False
tokenization.make_vocab_size_divisible_by = 128

# my_config.py:
tokenization.tokenizer.do_lower_case = False
```

Tokenization config must contain a tokenizer(e.g., `BertTokenizer`). `append_eod` and `make_vocab_size_divisible_by` are not necessary. 

`make_vocab_size_divisible_by` is used for padding the vocab size to be divisible by this value. This is added for computational efficiency for tensor parallelism.

## Get the Default Config

You don't need to rewrite all contents in config every time. You can import a config file as a python file or use function [`get_config`](../libai.config.html#libai.config.get_config).

If you build LiBai from source, you can get all default config files in `configs/*`. Then you can import the config files as follows:

```python
# import config
from .common.models.bert import pretrain_model as model
from .common.models.graph import graph
from .common.train import train
from .common.optim import optim
from .common.data.bert_dataset import dataloader, tokenization

# modify it
train.train_iter = 100
...
```

If you install LiBai by `pip`, you can use `get_config` function to get all default ocnfig files as follows:

```python
from libai.config import get_config
# get config
model = get_config("common/models/bert.py").pretrain_model
graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim
dataloader = get_config("common/data/bert_dataset.py").dataloader
tokenization = get_config("common/data/bert_dataset.py").tokenization

# modify it
train.train_iter = 100
...
```

## LazyConfig Best Practices

1. Treat the configs you write as actual "code": Avoid copying them or duplicating them. Import the common parts between configs.
2. Keep the configs you write simple: Don't include keys that do not affect the experimental setting.
