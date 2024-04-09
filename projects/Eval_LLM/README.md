# LLM Evaluation

A tool for evaluating OneFlow models based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/)

## Environment

Follow this [Installation Instruction](https://libai.readthedocs.io/en/latest/tutorials/get_started/Installation.html) to install oneflow(1.0.0) and libai first. Conda is recommended.  
**Make sure you have python>=3.10 to run evaluation for GLM.**
Then run ```pip install -r ./projects/Eval_LLM/requirements.txt``` to install dependencies.

## Run Eval

### Set the parameters in ./projects/Eval_LLM/config.py

> pretrained_model_path: The path of your model weights, either huggingface weights or libai weights is ok.
> hf_tokenizer_path: The path of huggingface tokenizer.
> model_type: Type of your model, this argument is need for loading model. All choices are listed in ./projects/Eval_LLM/special_arguments.json
> model_weight_type: Whether your weights are huggingface weights or libai weights.
> eval_tasks: Tasks you want to evaluate you model on.
> batch_size_per_gpu: Batch size on a single gpu, if you want to accelerate you evaluation, set it larger. But this may lead to OOM error.

Tasks for Evaluation are listed [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).

### Run the following command to start eval
```
bash tools/infer.sh projects/Eval_LLM/main.py 1
```
Notice: The number stands for how many gpus you want to use.

If you want to eval GLM(ChatGLM), run this:
```
CHATGLM_HF_DIR=YOUR_MODEL_PATH bash tools/infer.sh projects/Eval_LLM/main.py 1
```

Notice: To run a model with 6B parameters, you are about to have VRAM more than 24GB. You can use tensor or pipeline parallel on multiple devices.

To know more about distributed inference: https://docs.oneflow.org/en/master/parallelism/04_launch.html

## Example of Eval Result
Using Llama2-7b
```
{'sciq': 
    {'acc,none': 0.794, 'acc_stderr,none': 0.012795613612786583, 'acc_norm,none': 0.707, 'acc_norm_stderr,none': 0.014399942998441271, 'alias': 'sciq'}, 
'lambada_openai': 
    {'perplexity,none': 28.778403569948463, 'perplexity_stderr,none': 1.0792474430271395, 'acc,none': 0.33980205705414324, 'acc_stderr,none': 0.006598757339311441, 'alias': 'lambada_openai'}, 
'gsm8k': 
    {'exact_match,strict-match': 0.001516300227445034, 'exact_match_stderr,strict-match': 0.0010717793485492675, 'exact_match,flexible-extract': 0.01061410159211524, 'exact_match_stderr,flexible-extract': 0.002822713322387704, 'alias': 'gsm8k'}
}
```