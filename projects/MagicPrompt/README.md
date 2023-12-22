# MagicPrompt

This project is a NLP text generate tasks, which based on `gpt2` model to generate prompt-texts for AI drawings, such as `stable-diffusion`. we provides a [pipeline](./pipeline.py) that can be directly used for inference, and you can also use your own dataset to `customize` the gpt2 with your own data.

## How to use pipeline

- Prepare the gpt2 checkpoint, If you don't have suitable checkpoint, you can use [OneFlow/MagicPrompt-Stable-Diffusion](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/libai/magicprompt/OneFlow-MagicPrompt-Stable_Diffusion.zip) or [Gustavosta/MagicPrompt-Stable-Diffusion](https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion?text=My+name+is+Merve+and+my+favorite).

- Adjust the parameters in the pipeline.py, and run it.

- Here is an example showing MagicPrompt, which combines the [oneflow version of diffusion](https://github.com/Oneflow-Inc/diffusers/wiki/How-to-Run-OneFlow-Stable-Diffusion):

```python
from projects.MagicPrompt.pipeline import MagicPromptPipeline
import oneflow as torch
from diffusers import OneFlowStableDiffusionPipeline


pipeline = MagicPromptPipeline(
    "/projects/MagicPrompt/configs/gpt_inference.py",
    model_path="path/to/gpt2-checkpoint",
    mode="huggingface",
)

text = ["a dog"]
output = pipeline(inputs=text)
if dist.is_main_process():
    print(output)

pipe = OneFlowStableDiffusionPipeline.from_pretrained(
    "prompthero/midjourney-v4-diffusion",
    use_auth_token=True,
)

pipe = pipe.to("cuda")
prompt = output[0]['generated_text']
with torch.autocast("cuda"):
    images = pipe(prompt).images
    for i, image in enumerate(images):
        image.save(f"result.png")

```
- Generated prompt: `a dog in a astronaut suit and luffy, intricate, luffy, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, luffy, unreal engine 5, 8 k, art by artgerm`
<div style="text-align: center;">
  <img src="https://user-images.githubusercontent.com/53039617/202831136-b44a37d2-a210-4eca-9fea-1a01976e92df.png" width="30%">
</div>


## How to customize the gpt2 with your own data

### 1. Prepare your own datasets

- Official dataset address: [https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts](https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts).

- If you want to customize datasts, please prepare the data in txt format, and use `projects/MagicPrompt/datasets/datasets.py` to process it.

- Use `tools/preprocess_data.py` to process the json files, you can refer [https://libai.readthedocs.io/en/latest/tutorials/basics/Preprocessing_Dataset.html](https://libai.readthedocs.io/en/latest/tutorials/basics/Preprocessing_Dataset.html).

```python
IMPL=mmap
KEYS=text

python tools/preprocess_data.py \
        --input path/to/test_sample_cn.json \
        --json-keys ${KEYS} \
        --vocab-file path/to/vocab.txt \
        --merges-file path/to/merges.txt
        --dataset-impl ${IMPL} \
        --tokenizer-name GPT2Tokenizer \
        --do-lower-case \
        --do-chinese-wwm \
        --split-sentences \
        --output-prefix magic_prompt_${IMPL} \
        --workers 4 \
        --log-interval 2
```

> You can directly get the processed dataset by running the following command: 
> `wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/libai/magicprompt/magicprompt.zip`


### 2. Training

```bash 
bash tools/train.sh tools/train_net.py projects/MagicPrompt/configs/gpt2_training.py 1
```
