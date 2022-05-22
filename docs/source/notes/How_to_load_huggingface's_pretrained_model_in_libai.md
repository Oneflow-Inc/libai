# How to load Huggingface's pretrained model in LiBai
In this tutorial, we will introduce to users how to instantiate a pretrained oneflow model from huggingface's pre-trained model configuration.

## Steps
1. Prepare pretrained model weights file.
- Huggingface's pretrained model weights file(`pytorch_model.bin`) can be downloaded from https://huggingface.co/models.
- OneFlow's pretrained model weights saved using [`oneflow.save()`].

2. Prepare config file.
- Config file(`config.json`) can be downloaded from https://huggingface.co/models.

3. Move the files to the folder. The file structure should be like:
```bash
$ tree my_pre_trained_model_directory
path/to/my_pre_trained_model_directory/
├── pytorch_model.bin or oneflow_model
└── vocab.json
```

## Start Loading
You can load pretrained BERT as following:
```python
import libai
from libai.models.utils.model_utils import LoadPretrainedBert
from libai.config.configs.common.models.bert import cfg

load_fucntion = LoadPretrainedBert(
    model=libai.models.BertModel,
    default_cfg=cfg,
    pretrained_model_path='path/to/my_pre_trained_model_directory'
    )
bert = my_class.load_model()
```
