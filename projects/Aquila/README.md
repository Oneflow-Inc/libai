

## Aquila
### 推理
- cuda

```bash
python projects/Aquila/pipeline.py --model_path=/root/models/Aquila-7B --mode=huggingface
```

- npu

```bash
python projects/Aquila/pipeline.py --model_path=/data0/hf_models/Aquila-7B --mode=huggingface --device=npu
```

- xpu

```bash
python projects/Aquila/pipeline.py --model_path=/root/models/Aquila-7B --mode=huggingface --device=xpu
```

### 训练 TBD
```bash
python projects/Aquila/utils/data_prepare.py
bash tools/train.sh tools/train_net.py projects/Aquila/configs/aquila_sft.py 1
```
