这个分支用来和Megatron对齐。



环境准备：

1. git clone libai，并切换到相应commit
2. git clone Megatron-LM，并切换到相应commit
3. 安装apex



数据集准备：

1. 暂时还没上传



环境配置：

1. 设置Megatron-LM需要的环境变量

```shell
export MASTER_ADDR=127.0.0.1; export MASTER_PORT=13245; export WORLD_SIZE=1; export RANK=0; export LOCAL_RANK=0;
```

2. 把libai和Megatron-LM的路径加入`PYTHONPATH`。
3. 修改libai中的数据集路径。
   1. `config/t5_compare_loss.py` 中的 `data.data_path` 和 `data.vocab_file`
   2. `configs/common/data/nlp_data.py` 中 `data` 变量的 `data_path` 和 `vocal_file` 
   3. `tests/t5_test/get_megatron_t5.py` 中的 `os.environ['VOCAB_FILE']` 以及 `sys.argv.extend` 中的 `--vocab-file` 后面的参数
4. 修改libai中读取模型权重的路径和Megatron-LM中读取模型权重的路径。





复现步骤：

1. 进入libai/tests/t5_test，执行python3 main.py，会进行一次inference的对齐，并在当前目录下生成 `flow_t5.f` 文件夹（oneflow模型的权重）和 `Megatron_t5.pth` （Megatron-LM）的权重
2. 修改 https://github.com/Oneflow-Inc/libai/blob/a4857125084092043973ea317a434537debe0733/libai/trainer/default.py#L227 为步骤1中flow_t5.f的路径，用给定的权重初始化要训练的oneflow模型
2. 修改 https://github.com/marigoold/Megatron-LM/blob/main/pretrain_t5.py#L81 为步骤1中`Megatron_t5.pth` 的路径，用给定的权重初始化要训练的Megatron模型
2. 在Megatron-LM目录下运行 `sh examples/pretrain_t5.sh` 训练Megatron模型，loss保存在Megatron_loss.txt内
2. 在libai目录下运行 `zsh tools/train.sh configs/t5_compare_loss.py 1` 训练oneflow模型，loss保存在of_bert_loss.txt内



一些踩坑：

1. 刚开始看optimizer部分，oneflow的是AdamW，Megatron的是Adam，我以为这二者不一样就擅自改了，结果Megatron得Adam有一个参数是adam_w_mode=True，实际上还是AdamW

   