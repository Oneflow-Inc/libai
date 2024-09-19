
### 推理

- cuda PASS

```bash
python projects/Qwen/pipeline.py --model_path=/root/models/Qwen1.5-7B-Chat --mode=huggingface
```

- npu PASS

```bash
python projects/Qwen/pipeline.py --model_path=/data0/hf_models/qwen2/Qwen1.5-7B-Chat --mode=huggingface --device=npu
```

- xpu PASS

```bash
python projects/Qwen/pipeline.py --model_path=/root/models/Qwen1.5-7B-Chat --mode=huggingface --device=xpu
```

### 训练 

- data preparation

```bash
python projects/Qwen/utils/data_prepare.py
```

- cuda PASS

```bash
export NUM_GPUS=8
python3 -m oneflow.distributed.launch \
    --nproc_per_node ${NUM_GPUS} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 12345 \
        tools/train_net.py --config-file=projects/Qwen/configs/qwen_sft.py \
            graph.enabled=True \
            train.input_placement_device="cuda" \
            train.dist.device_type="cuda" \
            train.dist.pipeline_parallel_size=${NUM_GPUS}
```
A100-PCIE-40GB x 4 OOM

- xpu OOM

```bash
export NUM_GPUS=1
python3 -m oneflow.distributed.launch \
    --nproc_per_node ${NUM_GPUS} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 12345 \
        tools/train_net.py --config-file=projects/Qwen/configs/qwen_sft.py \
            graph.enabled=False \
            train.input_placement_device="xpu" \
            train.dist.device_type="xpu" \
            train.dist.pipeline_parallel_size=${NUM_GPUS}
```

- npu 没有测，应该不行


