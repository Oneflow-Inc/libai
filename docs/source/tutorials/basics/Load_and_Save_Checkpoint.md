# Load and Save a Checkpoint in LiBai

Instead of directly using [`flow.save()`](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=save#oneflow.save) and [`flow.load()`](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.load#oneflow.load), LiBai provides the [`checkpoint module`](https://libai.readthedocs.io/en/latest/modules/libai.utils.html#module-libai.utils.checkpoint) to deal with the complex situations when saving/loading model.


Typically, you don't need to write the relative code to load/save weights trained from LiBai when using LiBai's `DefaultTrainer` and `LazyConfig`. For more details, see [Training & Evaluation in Command Line](https://libai.readthedocs.io/en/latest/tutorials/basics/Train_and_Eval_Command_Line.html) which introduces `weight load` and `resume training` settings in `config.py` or in command line for standard training.

Here we introduce how to load&save weights according to your custom needs. Suppose you have a model trained with LiBai.

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

The following code shows how to load/save model weights:
```python
from libai.utils.checkpoint import Checkpointer
from path.to.your.build_model import build_model

model = build_model(cfg.model)
# load model weights
Checkpointer(model).load(path_to_model) # path_to_model should be "output/finetune_qqp/model_final" 

# save model weights
checkpointer = Checkpointer(model, save_dir="output/")
checkpointer.save("model_999")  # save to output/model_999
```

You can also save other informations (e.g. `optim`, `scheduler`) other than model weights by using `checkpointer`. See [libai.utils.checkpoint](https://libai.readthedocs.io/en/latest/modules/libai.utils.html#module-libai.utils.checkpoint) for more details.