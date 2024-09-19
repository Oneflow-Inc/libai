### Baichuan
#### 推理

- cuda PASS
```bash
python projects/Baichuan/pipeline.py --mode=huggingface --model_path=/root/models/Baichuan2-7B-Chat
```

- xpu PASS
```bash
python projects/Baichuan/pipeline.py --mode=huggingface --device=xpu --model_path=/root/models/Baichuan2-7B-Chat
```

#### 训练
- cuda PASS
```bash
export NUM_GPUS=8
python3 -m oneflow.distributed.launch \
    --nproc_per_node ${NUM_GPUS} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 12345 \
        tools/train_net.py --config-file=projects/Baichuan/configs/baichuan_sft.py \
            graph.enabled=True \
            train.input_placement_device="cuda" \
            train.dist.device_type="cuda" \
            train.dist.pipeline_parallel_size=${NUM_GPUS}
```

- xpu OOM after 7 iterations
```bash
export NUM_GPUS=1
python3 -m oneflow.distributed.launch \
    --nproc_per_node ${NUM_GPUS} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 12345 \
        tools/train_net.py --config-file=projects/Baichuan/configs/baichuan_sft.py \
            graph.enabled=False \
            train.input_placement_device="xpu" \
            train.dist.device_type="xpu" \
            train.dist.pipeline_parallel_size=${NUM_GPUS}
```
