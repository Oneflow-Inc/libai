

## Aquila
### 推理
- cuda

```bash
python projects/Aquila/pipeline.py --model_path=/root/models/Aquila-7B --mode=huggingface
```

### 训练
-- data preparation
```bash
python projects/Aquila/utils/data_prepare.py
```
- cuda
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

- xpu
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
