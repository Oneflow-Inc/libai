# How to load pretrained model in LiBai
In this tutorial, we will introduce to users how to instantiate a pretrained oneflow model.

## Steps
1. Prepare pretrained model weights file, which can be the form of `OneFlow` or `HuggingFace`.
- `OneFlow`'s pretrained model weights saved using [`oneflow.save()`].
- `Huggingface`'s pretrained model weights file(`pytorch_model.bin`) can be downloaded from https://huggingface.co/models.

2. Prepare config file.
> The config file is required when loading the `HuggingFace` model.
> `OneFlow`'s config file can be import directly from `configs/common/models`.
- `Huggingface`'s config file(`config.json`) can be downloaded from https://huggingface.co/models.

3. The structure of the pretrained model folder should be like:
```bash
# OneFlow pretrained model
$ tree pretrained_model_dir
path/to/pretrained_model_dir/
 └── oneflow_model

# Huggingface pretrained model
$ tree pretrained_model_dir
path/to/pretrained_model_dir/
 ├── pytorch_model.bin
 └── config.json
```

## Start Loading
You can load pretrained BERT as following:
```python
import libai
from libai.models.utils import BertLoaderHuggerFace, BertLoaderLiBai
from libai.config.configs.common.models.bert import cfg


# load huggingface weight
loader = BertLoaderHuggerFace(
    model=libai.models.BertModel,
    libai_cfg=cfg,
    pretrained_model_path='path/to/my_pretrained_model_directory',
    hidden_dropout_prob=0,
)
bert = loader.load()

# load libai weight
loader = BertLoaderLiBai(
    model=libai.models.BertModel,
    libai_cfg=cfg,
    pretrained_model_path='path/to/my_pretrained_model_directory',
    hidden_dropout_prob=0,
)
bert = loader.load()
```


# Use Custom ModelLoader

## Model Loader for HuggerFace
If you want to define your own HuggerFace's model loader, you can inherit the base `ModelLoaderHuggerFace` in `libai.models.utils.model_utils.base_loader`.

Then you need to overwrite the `_convert_state_dict` and `_load_config_from_json` method to load HuggingFace's pretrained model in LiBai. 

Finally, you need set `base_model_prefix_1` and `base_model_prefix_2` argument, which represent the base model name for HuggingFace and LiBai respectively.

## Model Loader for LiBai
If you want to define your own LiBai's model loader, you can inherit the base `ModelLoaderLiBai` class in `libai.models.utils.model_utils.base_loader`.

You just need to set `base_model_prefix_2` argument to load LiBai's pretrained model.