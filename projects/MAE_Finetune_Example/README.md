## MAE Finetune Example

This project is a practical example about how to finetune a ViT model that pretrained using MAE, based on [LiBai](https://libai.readthedocs.io/en/latest/)'s [MAE](https://github.com/Oneflow-Inc/libai/tree/main/projects/MAE) project. The finetuning is performed on [Stanford Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) dataset. This belongs to a fine-grained image classification task. Each category in this dataset has three kinds of information: make, model and year such as "AM General Hummer SUV 2000".

## Usage

### Prepare the Data

We provide a convenient shell script `prepare_on_onecloud.sh` to complete all of the preparation work, under the environment of [OneCloud](https://oneflow.cloud/#/). 

Just enter the root directory of this project, and execute:

```bash
bash prepare_on_onecloud.sh
```

Or you can modifiy the shell script and execute it in other environment.


### Prepare the Pretrained Model Checkpoint

You can download pretrained model checkpoints on [the official PyTorch implementation of MAE](https://github.com/facebookresearch/mae). And specify its path in "configs/mae_finetune_StanfordCars.py" of this project:

```python
finetune.path = "path/to/checkpoint"
```

### Finetuning

Enter the "projects" directory of LiBai, and execute:

```bash
python MAE/train_net.py --config-file MAE_Finetune_Example/configs/mae_finetune_StanfordCars.py
```

### Infer

Enter the root directory of this project, and execute:

```bash
python infer.py --config_file <path to config> --model_file <path to model> --image_file <path to input image>
```
