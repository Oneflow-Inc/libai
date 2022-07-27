# How to load pretrained model in LiBai
In this tutorial, we will introduce to users how to instantiate a pretrained oneflow model.

## Steps
1. Prepare pretrained model weights file, which can be the form of `OneFlow` or `Transfomers`.
- `OneFlow`'s pretrained model weights saved using [`oneflow.save()`].
- `Huggingface`'s pretrained model weights file(`pytorch_model.bin`) can be downloaded from https://huggingface.co/models.

2. Prepare config file.
> The config file is required when loading the `Transformers` model.
> When loading OneFlow config file, only need to import it from `configs/common/models`.
- Config file(`config.json`) can be downloaded from https://huggingface.co/models.

3. Move the files to the folder. The file structure should be like:
```bash
# Load OneFlow model
$ tree pretrained_model_dir
path/to/pretrained_model_dir/
 └── oneflow_model

# Load Transformers model
$ tree pretrained_model_dir
path/to/pretrained_model_dir/
 ├── pytorch_model.bin
 └── config.json
```

## Start Loading
You can load pretrained BERT as following:
```python
import libai
from libai.models.utils import BertLoaderHuugerFace, BertLoaderLiBai
from libai.config.configs.common.models.bert import cfg


# load huggingface weight
loader = BertLoaderHuugerFace(
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
