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

```
[09/19 14:39:40 lb.utils.events]:  eta: 22:07:15  iteration: 87/18660  consumed_samples: 704  total_loss: 10.36  time: 4.2893 s/iter  data_time: 0.0105 s/iter total_throughput: 1.87 samples/s lr: 6.99e-07
[09/19 14:39:44 lb.utils.events]:  eta: 22:07:07  iteration: 88/18660  consumed_samples: 712  total_loss: nan  time: 4.2889 s/iter  data_time: 0.0104 s/iter total_throughput: 1.87 samples/s lr: 7.07e-07
NaN or Inf found in input tensor.
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
