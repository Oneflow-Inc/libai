## Introduction
Here are the Weight Loaders currently supported in LiBai. You can use them to load the models in LiBai and the models stored on the huggingface.


## Weight Loader On LiBai
- [BERT Loader](./bert_loader.py)
- [RoBERTa Loader](./roberta_loader.py)
- [GPT2 Loader](./gpt_loader.py)
- [MT5 Loader](../../../../projects/MT5/utils/mt5_loader.py)
- [SWIN Loader](./swin_loader.py)
- [SWIN2 Loader](./swinv2_loader.py)
- [VIT Loader](./vit_loader.py)

## How To Use
We can easily load pretrained BERT as following:
```python
import libai
from libai.models.utils import BertLoaderHuggerFace, BertLoaderLiBai
from configs.common.models.bert import cfg

# load huggingface weight
loader = BertLoaderHuggerFace(
    model=libai.models.BertModel,
    libai_cfg=cfg,
    pretrained_model_path="path/to/huggingface_pretrained_model_directory",
    hidden_dropout_prob=0,
    apply_residual_post_layernorm=True
)
bert = loader.load()

# load libai weight
loader = BertLoaderLiBai(
    model=libai.models.BertModel,
    libai_cfg=cfg,
    pretrained_model_path='path/to/libai_pretrained_model_directory',
    hidden_dropout_prob=0,
)
bert = loader.load()
```
