# ChatGLM
Reproduce ChatGLM with OneFlow, which effect are equivalent to HuggingFace's [ChatGLM3](https://huggingface.co/THUDM/chatglm3-6b).

## Introduce
The ChatGLM Supervised FineTuning project can support 3D parallel.

## FineTuning ChatGLM3
FineTuning ChatGLM3 on 8 GPUs using parallelism.

### 1. Prepare the sft dataset
#### download dataset
```bash
export DATA_DIR=~/DATA # [At the beginning, it was an empty folder]
cd $DATA_DIR
git clone https://www.modelscope.cn/datasets/DAMO_ConvAI/ZhDoc2BotDialogue.git
```

#### preprocess
```bash
cd projects/ChatGLM
python utils/prepare_CoT_zh.py
```
### 2. Prepare your finetuning config file

> set the finetuning parameters in `projects/ChatGLM/configs/chatglm_sft.py`, such as `dataset_path` and `pretrained_model_path`.


### 3. Run the following code to start SFT
```bash
# cd /path/to/libai
bash tools/train.sh projects/ChatGLM/train_net.py projects/ChatGLM/configs/chatglm_sft.py 8
```

## ChatGLM Inference

- Prepare the ChatGLM checkpoint.
- Adjust the parameters in the `projects/ChatGLM/pipeline.py`, and running:
```bash
python projects/ChatGLM/pipeline.py
```