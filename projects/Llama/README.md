# Llama2

Reproduce Llama2 with OneFlow, which effect are equivalent to HuggingFace's [Llama2](https://huggingface.co/docs/transformers/v4.35.2/en/model_doc/llama2#overview).

## Introduce
The Llama2 Supervised FineTuning project can support 3D parallel.

## FineTuning Llama2
FineTuning llama2 on 8 GPUs using parallelism.

### 1. Prepare the alpaca dataset

> set the parameters in `projects/Llama/utils/prepare_alpaca.py` for prepare the datasets, such as `destination_path` and `checkpoint_dir`.

> Get the alpaca dataset files by running:
```python3
# path/to/libai
python projects/Llama/utils/prepare_alpaca.py
```

### 2. Prepare your finetuning config file

> set the finetuning parameters in `projects/Llama/configs/llama_sft.py`, such as `dataset_path` and `pretrained_model_path`.

### 3. Run the following code to start SFT
```bash
# full finetune
bash tools/train.sh projects/Llama/train_net.py projects/Llama/configs/llama_sft.py 8

# adapter finetune
bash tools/train.sh projects/Llama/adapter/train_net.py projects/Llama/adapter/adapter_sft.py 8
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