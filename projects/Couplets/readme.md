## Couplets in LiBai

Contributor{Yulin Zhuang: [https://github.com/ZylOo0](https://github.com/ZylOo0)}

This is the LiBai implementation of Couplets

## Supported parallel mode and task
Based on [libai.layers](https://libai.readthedocs.io/en/latest/modules/libai.layers.html), Couplets model is automatically configured with the following parallelism mode.

<table class="docutils">
  <tbody>
    <tr>
      <th width="80"> Model </th>
      <th valign="bottom" align="left" width="120">Data Parallel</th>
      <th valign="bottom" align="left" width="120">Tensor Parallel</th>
      <th valign="bottom" align="left" width="120">Pipeline Parallel</th>
    </tr>
    <tr>
      <td align="left"> <b> Couplets training </b> </td>
      <td align="left">&#10004;</td>
      <td align="left">&#10004;</td>
      <td align="left">&#10004;</td>
    </tr>
    <tr>
      <td align="left"> <b> Couplets inference </b> </td>
      <td align="left"> - </td>
      <td align="left">&#10004;</td>
      <td align="left">&#10004;</td>
    </tr>
  </tbody>
</table>

## Setup env

Install in LiBai, refer to [LiBai install doc](https://libai.readthedocs.io/en/latest/tutorials/get_started/Installation.html)

## Prepare Datasets

Download datasets and unzip data:
```shell
wget http://oneflow-static.oss-cn-beijing.aliyuncs.com/libai/couplets/couplets.zip
unzip couplets.zip
```

you will get dataset like this:
```
couplets
├── test
│   ├── in.txt
│   └── out.txt
├── train
│   ├── in.txt
│   └── out.txt
└── vocab.txt
```

## Training

- Set dataset path in `configs/config.py`

    ```python
    dataloader.train = LazyCall(build_nlp_train_loader)(
        dataset=[
            LazyCall(CoupletsDataset)(
                path="data_test/couplets", # set to your data_path
                is_train=True,
                maxlen=64,
            )
        ],
        num_workers=4,
    )
    dataloader.test = [
        LazyCall(build_nlp_test_loader)(
            dataset=LazyCall(CoupletsDataset)(
                path="data_test/couplets", # set to your data_path
                is_train=False,
                maxlen=64,
            ),
            num_workers=4,
        )
    ]
    ```
- set model cfg in `configs/config.py` according to your needs
    ```python
    transformer_cfg = dict(
        vocab_size=9027,
        max_position_embeddings=64,
        hidden_size=512, # modify it according to your needs
        intermediate_size=512, # modify it according to your needs
        hidden_layers=6, # modify it according to your needs
        num_attention_heads=8, # modify it according to your needs
        embedding_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        initializer_range=0.02,
        layernorm_epsilon=1e-5,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=True,
    )
    ```
- set dist config in `configs/config.py` according to your needs, refer to [LiBai distribute doc](https://libai.readthedocs.io/en/latest/tutorials/basics/Distributed_Configuration.html) for more details
    ```python
    dist=dict(
        data_parallel_size=1, # modify it according to your needs
        tensor_parallel_size=1, # modify it according to your needs
        pipeline_parallel_size=1, # modify it according to your needs
        pipeline_stage_id=None, # modify it according to your needs
        pipeline_num_layers=model.cfg.hidden_layers * 2,
    ),
    ```

- Following [quick_run](https://libai.readthedocs.io/en/latest/tutorials/get_started/quick_run.html) in LiBai, run training command in LiBai **root** dir
    ```shell
    # cd path to libai
    bash tools/train.sh tools/train_net.py projects/Couplets/configs/config.py 4
    ```

- After finish training, you will get trained model in path `output/couplet`
    ```shell
    output/couplet
    ├── config.yaml
    ├── last_checkpoint
    ├── log.txt
    ├── log.txt.rank1
    ├── log.txt.rank2
    ├── log.txt.rank3
    ├── metrics.json
    ├── model_0004999
    ├── model_0009999
    ├── model_0014999
    ├── model_0019999
    ├── model_0024999
    └── model_final
    ```

## inference

- for inference in one gpu: 
    ```
    # modify path in projects/Couplets/infer.py
    # config_file = "output/couplet/config.yaml"
    # checkpoint_file = "output/couplet/model_final"
    # vocab_file = "data_test/couplets/vocab.txt"
    
    python projects/Couplets/infer.py
    ```

- for distributed inference:
    
    set your data path and model in `projects/Couplets/distribute_infer.py`
    ```python
    # line 46
    self.cfg.vocab_file = "data_test/couplets/vocab.txt"

    # line 97 ~106
    pipeline = CoupletPipeline(
        # you can also use path output/couplet/config.yaml to replace config.py 
        "projects/Couplets/configs/config.py",
        data_parallel=1,
        tensor_parallel=1, # modify it according to your needs
        pipeline_parallel=4, # modify it according to your needs
        pipeline_stage_id=None, # modify it according to your needs
        pipeline_num_layers=12, # modify it according to your needs
        model_path="output/couplet/model_final/model", # modify it according to your needs
        mode="libai",
    )
    ```
    ```
    bash tools/infer.sh projects/Couplets/distribute_infer.py 4
    ```

## Results

```shell
上联：
天朗气清风和畅
下联：
水流花海月圆融

上联：
千秋月色君长看
下联：
一夜风流人在天

上联：
梦里不知身是客
下联：
此间何处是家乡
```