# Llama2

Reproduce Llama2 with OneFlow, which effect are equivalent to HuggingFace's [Llama2](https://huggingface.co/docs/transformers/v4.35.2/en/model_doc/llama2#overview).

## Introduce
The Llama2 Supervised FineTuning project can support 3D parallel.

## FineTuning Llama2
FineTuning llama2 on 8 GPUs using parallelism.

### 1. Prepare the alpaca dataset

> Alpaca Dataset address: https://huggingface.co/datasets/vicgalle/alpaca-gpt4

### 2. Prepare your finetuning config file

> set the finetuning parameters in `projects/Llama/configs/llama_sft.py`, such as `dataset_path` and `pretrained_model_path`.

### 3. Run the following code to start SFT
```bash
# cd /path/to/libai
bash tools/train.sh projects/Llama/train_net.py projects/Llama/configs/llama_sft.py 8
```

## Evaluate

> set the eval parameters in `/data/home/xiezipeng/libai/projects/Llama/utils/eval_adapter.py`, and running:
```python3
python projects/Llama/utils/eval_adapter.py
```

## Llama2 Inference

- Prepare the Llama2 checkpoint.
- Adjust the parameters in the `projects/Llama/pipeline.py`, and running:
```bash
bash tools/infer.sh projects/Llama/pipeline.py 8
```