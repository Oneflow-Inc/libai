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

**train**

**optim**

**dataloader**

**tokenization (optional)**



#### Get the Default Config

You do not need to rewrite all contents in a config file every time, you can 



#### Best Practice with LazyConfig

