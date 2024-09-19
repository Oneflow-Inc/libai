
### 推理

- cuda TODO

```bash
python projects/Qwen/pipeline.py --model_path=/root/models/Qwen1.5-7B-Chat --mode=huggingface
```

- npu

```bash
python projects/Qwen/pipeline.py --model_path=/data0/hf_models/qwen2/Qwen1.5-7B-Chat --mode=huggingface --device=npu
```

- xpu

```bash
python projects/Qwen/pipeline.py --model_path=/root/models/Qwen1.5-7B-Chat --mode=huggingface --device=xpu
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


