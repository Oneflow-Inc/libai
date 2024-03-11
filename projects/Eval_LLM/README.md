# LLM Evaluation

Evaluation with [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness/) powered by [transformers](https://github.com/huggingface/transformers) and oneflow.

## Supported Models

Bloom  
Chatglm  
Llama  

## Eval

### Environment

Follow this [Installation Instruction](https://libai.readthedocs.io/en/latest/tutorials/get_started/Installation.html) to install oneflow and libai first. Conda is recommended.  
**Make sure you have python>=3.10 to run evaluation for chatglm.**
Then run ```pip install -r ./projects/Eval_LLM/requirements.txt``` to install dependencies.

### Run Eval

#### Set the parameters in ./projects/Eval_LLM/config.py

Tasks for Evaluation are listed [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.3.0/lm_eval/tasks)

#### Run the following command to start eval
```
python -m oneflow.distributed.launch --nproc_per_node 4 ./projects/Eval_LLM/main.py
```

If you want to eval ChatGLM, run this:
```
CHATGLM_HF_DIR=YOUR_MODEL_PATH  python -m oneflow.distributed.launch --nproc_per_node 4 ./projects/Eval_LLM/main.py
```

Attention: To run a model with 6B parameters, you are about to have VRAM more than 24GB. You can use tensor or pipeline parallel on multiple devices.

To know more about distributed inference: https://docs.oneflow.org/en/master/parallelism/04_launch.html