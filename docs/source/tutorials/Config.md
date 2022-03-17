### Lazy Configs

We find the traditional yacs-based config system or python argparse command-line options cannot offer enough flexibility for new project development. We just borrowed the [lazy config system](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) from detectron2 as an alternative, non-intrusive config system for LiBai.

You can read the [d2 tutorial](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) for the syntax and basic usage of lazy config. Here we will show you some example usage in LiBai.

#### Configs in LiBai

In LiBai, we define a standard set of config namespace for later use. This set of namespace must be kept if you want to use complete training and evaluation process of LiBai. 

In summary, this namespace is `model, graph, train, optim, dataloader, tokenization(optional)`, and we will introduce it in detail below.

**model**

This is the config for model definition. You can see some examples in `configs/common/models`.

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

After you define the model config in a python file, you can `import` it in the global scope of the config file. Note that you need to rename it as `model` regardless of the name used in model config.

You can access and change all keys in the model config after import.

**graph**

This is the config for `nn.Graph` mode. You can learn more information about the static graph mode in official [nn.Graph docs](https://docs.oneflow.org/master/basics/08_nn_graph.html).

LiBai has already defined a `GraphBase` class for almost all models use. You can simply turn on this option converting eager mode to graph mode. 

The graph config can be found in `configs/common/models/graph.py`, and two useful options are shown as follows:

```python
# Turn on graph mode, if set to `False`, will use eager mode.
graph.enabled = True 

# Set graph debug level, -1 means no debug info, and 0,1,2,3 can be 
# set for different debug levels. 
# More information can be found in nn.Graph documents.
graph.debug = -1 
```

**train**

This is the config for training and evaluation. You can find the default train config in `configs/common/train.py`.

We will show you the convention about training / test specific parameters as follows:

```python
train = dict(
    
    # Directory where output files are written
    output_dir="./output",

    # `train_micro_batch_size` is number of images per batch on each GPU. 
    # train_mini_batch_size = train_micro_batch_size * num_accumulation_steps.
    # This is also the number of training images per step (i.e. per iteration). 

    # If we use 8 GPUs for data parallel groups, `train_micro_batch_size = 2` and 
    # `num_accumulation_steps = 4`, then each GPU will see 2 images per batch and
    # 8 images per iteration.
    # Total 64 images will be trained per iteration across all GPUs.

    # global_batch_size = micro_batch_size  * num_grad_acc * data_parallel_groups
    train_micro_batch_size=32,
    global_batch_size=None,
    num_accumulation_steps=None,

    # The total training iteration
    train_iter=10000,
    # The total training epoch, will be scaled to iteration automatically.
    # We will choose by `max(train_iter, train_epoch * iter_per_epoch)`.
    train_epoch=0,  
    consumed_train_samples=0,
    consumed_valid_samples=0,
    train_samples=None,

    # Fraction of lr-warmup-iters to use for warmup (as a float)
    warmup_ratio=0,  

    # The start iteration, usually needn't set it manually.
    # It can be computed automatically when resuming training.
    start_iter=0,

    # Enable automatic mixed precision for training
    # Note that this does not change model's inference behavior.
    amp=dict(enabled=False),  

    # Enable activation checkpointing to allow for training
    # with larger models, sequences, and batch sizes.
    # Checkpoint the input activations of each transformer layers by default.
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
    
    # Save a checkpoint after every this number of iterations,
    # and maximum number of checkpoint will be kept.
    checkpointer=dict(period=5000, max_to_keep=100),  

    # Options for evaluation

    # `test_micro_batch_size` is number of images per batch on each GPU for testing. 
    # If we use 8 GPUs for data parallel groups and `test_micro_batch_size = 2`, then
    # total 16 images will be used per iteration across all GPUs.
    test_micro_batch_size=32,

    # Enabled evaluation during training every `eval_period` number of iterations.
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

    # Path to a checkpoint file to be loaded to the model. 
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
    # See https://libai.readthedocs.io/en/latest/tutorials/Getting%20Started.html for more detail.
    dist=dict(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    ),
    
    # Set seed to positive to use a fixed seed. Note that a fixed seed increases
    # reproducibility but does not guarantee fully deterministic behavior.
    # Disabling all parallelism further increases reproducibility.
    seed=1234,
)
```

**optim**

This is 

**dataloader**

**tokenization (optional)**



#### Get the Default Config

You do not need to rewrite all contents in a config file every time, you can 



#### Best Practice with LazyConfig

