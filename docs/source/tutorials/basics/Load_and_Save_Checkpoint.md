# Load/Save a Checkpoint in LiBai

Instead of directly using [`flow.save()`](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=save#oneflow.save) and [`flow.load()`](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.load#oneflow.load), LiBai provides a [`checkpoint module`](https://libai.readthedocs.io/en/latest/modules/libai.utils.html#module-libai.utils.checkpoint) dealing with complex situations for saving/loading model.


Typically, users don't need to write load/save weights trained from LiBai when using LiBai's `DefaultTrainer` and `LazyConfig`.[Training & Evaluation in Command Line](https://libai.readthedocs.io/en/latest/tutorials/basics/Train_and_Eval_Command_Line.html) introduces `load weight` and `resume training` settings in `config.py` or in command line for standard training.

Here we introduce how to load/save weights for your custom usage, suppose you have a model trained with LiBai

```shell
# your model directory
output/finetune_qqp
├── config.yaml
├── last_checkpoint
├── log.txt
├── log.txt.rank1
├── log.txt.rank2
├── log.txt.rank3
├── metrics.json
├── model_0000009
│   ├── graph
│   ├── lr_scheduler
│   └── model
├── model_0000019
│   ├── graph
│   ├── lr_scheduler
│   └── model
├── model_best
│   ├── graph
│   ├── lr_scheduler
│   └── model
└── model_final
    ├── graph
    ├── lr_scheduler
    └── model
```

Load/Save model weights in your code
```python
from libai.utils.checkpoint import Checkpointer

model = build_model(cfg.model)
# load model weights
Checkpointer(model).load(path_to_model) # path_to_model should be "output/finetune_qqp/model_final" 

# save model weights
checkpointer = Checkpointer(model, save_dir="output/")
checkpointer.save("model_999")  # save to output/model_999
```

You can also save more information(e.g. `optim`, `scheduler`) by Using `checkpointer`, see [Api doc](https://libai.readthedocs.io/en/latest/modules/libai.utils.html#module-libai.utils.checkpoint) for more details.