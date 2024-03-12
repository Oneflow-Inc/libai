# LLM Evaluation

Evaluation with [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness/) powered by [transformers](https://github.com/huggingface/transformers) and oneflow.

## Supported Models

Bloom  
GLM(ChatGLM)  
Llama  

## Eval

### Environment

Follow this [Installation Instruction](https://libai.readthedocs.io/en/latest/tutorials/get_started/Installation.html) to install oneflow and libai first. Conda is recommended.  
**Make sure you have python>=3.10 to run evaluation for GLM.**
Then run ```pip install -r ./projects/Eval_LLM/requirements.txt``` to install dependencies.

### Run Eval

#### Set the parameters in ./projects/Eval_LLM/config.py

Tasks for Evaluation are listed [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.3.0/lm_eval/tasks)

#### Run the following command to start eval
```
python ./projects/Eval_LLM/main.py
```
Or on multiple devices
```
python -m oneflow.distributed.launch --nproc_per_node 4 ./projects/Eval_LLM/main.py
```

If you want to eval GLM(ChatGLM), run this:
```
CHATGLM_HF_DIR=YOUR_MODEL_PATH  python -m oneflow.distributed.launch --nproc_per_node 4 ./projects/Eval_LLM/main.py
```

Attention: To run a model with 6B parameters, you are about to have VRAM more than 24GB. You can use tensor or pipeline parallel on multiple devices.

To know more about distributed inference: https://docs.oneflow.org/en/master/parallelism/04_launch.html

### Example of Eval Result
Using Llama2-7b
```
{'results': 
    {'hellaswag': 
        {'acc': 0.5672176857199761, 'acc_stderr': 0.004944485990639519, 'acc_norm': 0.7294363672575184, 'acc_norm_stderr': 0.004433430790349411}, 
    'lambada_openai': 
        {'ppl': 1414021.2239531504, 'ppl_stderr': 68163.8970088081, 'acc': 0.0, 'acc_stderr': 0.0}, 
    'sciq': 
        {'acc': 0.904, 'acc_stderr': 0.009320454434783205, 'acc_norm': 0.687, 'acc_norm_stderr': 0.014671272822977885}
    },
    'versions': 
        {'hellaswag': 0, 'lambada_openai': 0, 'sciq': 0}, 
    'config': 
        {'model': 'llama', 'batch_size': 4, 'device': 'cuda:0', 'num_fewshot': 0, 'limit': None, 'bootstrap_iters': 100000}
}
```