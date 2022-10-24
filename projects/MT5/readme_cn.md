# MT5 模型文档

## 环境配置

1. 安装最新的 [OneFlow](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package)，支持 `Python` 版本 `3.7, 3.8, 3.9, 3.10`。

```bash
python3 -m pip install --upgrade pip #--user
```

```bash
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu112
```

2. 从 github 获取 [Libai](https://github.com/Oneflow-Inc/libai) 库

```bash
git clone --recursive https://github.com/Oneflow-Inc/libai.git
```

或

```bash
git clone --recursive git@github.com:Oneflow-Inc/libai.git
```

然后进入到 `libai` 工程目录下并切换到 `dev_optimize_MT5` 分支


```bash
cd libai
git checkout dev_optimize_MT5
```

3. 安装 Libai

在 `libai` 根目录下运行以下命令：

```bash
pip install -r requirements.txt # 安装依赖项
pip install -e . # 以可编辑模式安装 Libai，这样对于 Libai 库核心功能的修改都可以直接生效
```

## MT5 训练流程

MT5模型目录: `PATH_TO_LIBAI_ROOT/projects/MT5`

### 下载demo训练数据

在 `libai` 项目的根目录下运行以下命令:

```bash
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/bert-base-chinese-vocab.txt -P ./data_test/bert_data/
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.bin -P ./data_test/bert_data/
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.idx -P ./data_test/bert_data/
```

如果是准备自己的数据集可以参考文档：`https://libai.readthedocs.io/en/latest/tutorials/basics/Preprocessing_Dataset.html`

### 运行单机单卡训练

在 `libai` 项目的根目录下运行以下命令：

```bash
NODE=1 NODE_RANK=0 ADDR=192.168.30.21 PORT=12345 bash tools/train.sh tools/train_net.py projects/MT5/configs/mt5_pretrain.py 1
```

A100 测试数据：
```bash
time: ~0.23 s/iter 
total_throughput: ~70 samples/s
```

### 运行单机8卡数据并行训练
在 `libai` 项目的根目录下运行以下命令：

```bash
NODE=1 NODE_RANK=0 ADDR=192.168.30.21 PORT=12345 bash tools/train.sh tools/train_net.py projects/MT5/configs/mt5_pretrain.py 8
```

A100 测试数据：
```bash
time: ~0.3 s/iter
total_throughput: ~426 samples/s
```

### 运行单机8卡张量并行

修改 `PATH_TO_LIBAI_ROOT/projects/MT5/configs/mt5_pretrain.py` 文件 `41~43` 行如下：

```python3
dist=dict(
    data_parallel_size=1,
    tensor_parallel_size=8,
    pipeline_parallel_size=1,
    pipeline_num_layers=2 * model.cfg.hidden_layers,
),
```

在 `libai` 项目的根目录下运行以下命令：

```bash
NODE=1 NODE_RANK=0 ADDR=192.168.30.21 PORT=12345 bash tools/train.sh tools/train_net.py projects/MT5/configs/mt5_pretrain.py 8
```

A100 测试数据：
```bash
time: ~0.3 s/iter
total_throughput: ~426 samples/s
```

### 运行单机8卡流水并行

在 `libai` 项目的根目录下运行以下命令：

```bash
NODE=1 NODE_RANK=0 ADDR=192.168.30.21 PORT=12345 bash tools/train.sh tools/train_net.py projects/MT5/configs/mt5_pretrain.py 8
```


