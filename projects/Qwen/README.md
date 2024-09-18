
### 推理

- cuda TODO

```bash
python projects/Qwen/pipeline.py --model_path=/root/models/Qwen1.5-7B-Chat --mode=huggingface
```

- npu TODO

```bash
python projects/Qwen/pipeline.py --model_path=/data0/hf_models/qwen2/Qwen1.5-7B-Chat --mode=huggingface --device=npu
```

- xpu


```bash
python projects/Qwen/pipeline.py --model_path=/root/models/Qwen1.5-7B-Chat --mode=huggingface --device=xpu
```

目前报错:
```bash
Traceback (most recent call last):
  File "projects/Qwen/pipeline.py", line 144, in <module>

  File "/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/click/core.py", line 1126, in __call__
    return self.main(*args, **kwargs)
  File "/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/click/core.py", line 1051, in main
    rv = self.invoke(ctx)
  File "/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/click/core.py", line 1393, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/click/core.py", line 752, in invoke
    return __callback(*args, **kwargs)
  File "projects/Qwen/pipeline.py", line 139, in main
    text = ["给出3点关于保持身体健康的意见。"]
  File "/workspace/git-repos/libai/libai/inference/basic.py", line 180, in __call__
    outputs_dict = self.postprocess(model_outputs_dict, **postprocess_params)
  File "projects/Qwen/pipeline.py", line 88, in postprocess
    def postprocess(self, model_output_dict, **kwargs) -> dict:
  File "projects/Qwen/pipeline.py", line 89, in <listcomp>
    return_ids = model_output_dict["return_ids"]
  File "/workspace/git-repos/libai/projects/Qwen/tokenizer.py", line 202, in decode
    token_ids,
  File "/workspace/git-repos/libai/libai/tokenizer/tokenization_base.py", line 930, in decode
    sub_texts.append(token)
  File "/workspace/git-repos/libai/projects/Qwen/tokenizer.py", line 190, in convert_tokens_to_string
    def _convert_id_to_token(self, index):
TypeError: sequence item 85: expected str instance, NoneType found
```

### 训练 TODO

- data preparation

```bash
python projects/Aquila/utils/data_prepare.py
```

- cuda 通过

```bash
export NUM_GPUS=4
python3 -m oneflow.distributed.launch \
    --nproc_per_node ${NUM_GPUS} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 12345 \
        tools/train_net.py --config-file=projects/Aquila/configs/aquila_sft.py \
            graph.enabled=True \
            train.input_placement_device="cuda" \
            train.dist.device_type="cuda" \
            train.dist.pipeline_parallel_size=${NUM_GPUS}
```

- xpu iter 0 通过, 然后就OOM

```bash
export NUM_GPUS=1
python3 -m oneflow.distributed.launch \
    --nproc_per_node ${NUM_GPUS} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 12345 \
        tools/train_net.py --config-file=projects/Aquila/configs/aquila_sft.py \
            graph.enabled=True \
            train.input_placement_device="xpu" \
            train.dist.device_type="xpu" \
            train.dist.pipeline_parallel_size=${NUM_GPUS}
```

- npu 没有测，应该不行


