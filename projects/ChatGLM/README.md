# ChatGLM
Reproduce ChatGLM with OneFlow, which effect are equivalent to HuggingFace's [ChatGLM3](https://huggingface.co/THUDM/chatglm3-6b).

## Introduce
The ChatGLM Supervised FineTuning project can support 3D parallel.

## FineTuning ChatGLM3
FineTuning ChatGLM3 on 8 GPUs using parallelism.

### 1. Prepare environment variables
```bash
export DATA_DIR=~/DATA/alpaca # [At the beginning, it was an empty folder]
export CHATGLM_HF_DIR=modelscope/hub/ZhipuAI/chatglm3-6b # [Your ChatGLM huggingface path]
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # [if need]
```

### 2. Prepare the sft dataset
```bash
cd projects/ChatGLM
python utils/prepare_data_alpaca.py
```

### 3. Run the following code to start SFT
```bash
# cd /path/to/libai
bash tools/train.sh projects/ChatGLM/train_net.py projects/ChatGLM/configs/chatglm_sft.py 8
```

## ChatGLM Inference
- Prepare the ChatGLM checkpoint.
- Adjust the parameters in the `projects/ChatGLM/pipeline.py`, and running:
### dp mode
```bash
bash tools/infer.sh projects/ChatGLM/pipeline.py 4
```
### naive mode
```bash
python projects/ChatGLM/pipeline.py
```

## Lora Part
 ![lora_finetune](./images/lora_finetune.svg) 

### ChatGLM Lora Finetune

- set `projects/ChatGLM/configs/chatglm_config.py`, lora_enable=True, same step with no lora.

### ChatGLM Lora Inference
- set `projects/ChatGLM/configs/chatglm_config.py`, lora_enable=True, same step with no lora.
