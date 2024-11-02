# Llama3

Reproduce Llama3 with OneFlow, which effect are equivalent to HuggingFace's [Llama3](https://huggingface.co/docs/transformers/main/en/model_doc/llama3#overview).

## Introduce
The Llama3 Supervised FineTuning project can support 3D parallel.

## FineTuning Llama3
FineTuning Llama3 on 8 GPUs using parallelism.

### 1. Prepare the alpaca dataset

> set the parameters in `projects/Llama3/utils/prepare_alpaca.py` for prepare the datasets, such as `destination_path` and `checkpoint_dir`.

> Get the alpaca dataset files by running:
```python3
# path/to/libai
python projects/Llama3/utils/prepare_alpaca.py
```

### 2. Prepare your finetuning config file

> set the finetuning parameters in `projects/Llama3/configs/llama_sft.py`, such as `dataset_path` and `pretrained_model_path`.

### 3. Run the following code to start SFT
```bash
# full finetune
bash tools/train.sh projects/Llama3/train_net.py projects/Llama3/configs/llama_sft.py 8

# adapter finetune
bash tools/train.sh projects/Llama3/adapter/train_net.py projects/Llama3/adapter/adapter_sft.py 8
```

## Evaluate

> set the eval parameters in `/data/home/xiezipeng/libai/projects/Llama3/utils/eval_adapter.py`, and running:
```python3
python projects/Llama3/utils/eval_adapter.py
```

## Llama3 Inference

- Prepare the Llama3 checkpoint.
- Adjust the parameters in the `projects/Llama3/pipeline.py`, and running:
```bash
bash tools/infer.sh projects/Llama3/pipeline.py 8
```

## npu/xpu example

- npu
```bash
python projects/Llama3/pipeline.py --device=npu --mode=huggingface --model_path /your/model/path
```

- xpu
```bash
python projects/Llama3/pipeline.py --device=xpu --mode=huggingface --model_path /your/model/path
```

