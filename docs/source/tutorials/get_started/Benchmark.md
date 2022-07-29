# Benchmarks

Here we provides our benchmark speed test results of LiBai's models compared with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) implementations. In LiBai V0.2.0, we only benchmark the speed tests under 32 GPUs in 4 nodes and all of the experiments were conducted under the same settings for a fair comparison.

## Settings
### Environments

- The commit of LiBai for comparison: [commit](https://github.com/Oneflow-Inc/libai/commit/9fc504c457da4fd1e92d854c60b7271c89a55222)
- The commit of OneFlow for comparison: [commit](https://github.com/Oneflow-Inc/oneflow/commit/55b822e4d3c88757d11077d7546981309125c73f)
- The commit of Megatron-LM for comparison: [commit](https://github.com/NVIDIA/Megatron-LM/commit/e156d2fea7fc5c98e645f7742eb86b643956d840)

### Model Hyper-parameters
- **BERT Model**
```python
num_layers = 24/48
num_attention_heads = 16
hidden_size = 1024
seq_length = 512
```
- **GPT-2 Model**
```python
num_layers = 24/48
num_attention_heads = 16
hidden_size = 1024
seq_length = 1024
```


## Main Results
Here we explain the evaluation indicators in the following tables:
- **fp16**: mixed precision training
- **nl**: num layers (When pipeline parallel size = 8, in order to have a relative number of layers per stage for computation, we adjust the num layers from 24 to 48)
- **ac**: enable activation checkpointing
- **mb**: micro-batch size per gpu
- **gb**: global batch size total
- **d x m x p**:
  - d: data-parallel-size
  - m: tensor-model-parallel-size
  - p: pipeline-model-parallel-size
- **1n1g**: 1 node, 1 gpu
- **2n8g**: 2 nodes, 8 gpus per node, 16 gpus in total
- **4n8g**: 4 nodes, 8 gpus per node, 32 gpus in total
- `grad_acc_num_step = global_batch_size / (micro_batch_size * data_parallel_size)`
- **samples/s**: throughput


### Data Parallel

<table class="docutils">
    <thead>
        <tr class="header">
            <th>BERT</th>
            <th>LiBai</th>
            <th>Megatron</th>
        </tr>
    </thead>
    <tbody>
        <tr class="odd">
            <td>nl24_fp16_1x1x1_mb24_gb24_1n1g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n1g/LibAI_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb24_gb24_1n1g_20220705_071307389288504/output.log">46.91</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n1g/Megatron_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb24_gb24_1n1g_20220615_130039677349789.log">42.6</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_8x1x1_mb16_gb64_1n4g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e_supple/1n4g/LibAI_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb64_1n4g_20220706_103618805733678/output.log">176.88</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base_supple/1n4g/Megatron_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb64_1n4g_20220706_121753217673018.log">154.7</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td>nl24_fp16_8x1x1_mb16_gb128_1n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb128_1n8g_20220705_101124804210475/output.log">351.57</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb128_1n8g_20220705_140535074517604.log">309.2</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_16x1x1_mb16_gb256_2n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb256_2n8g_20220705_172421459267607/output.log">675.87</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb256_2n8g_20220705_193107517518321.log">534.7</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td>nl24_fp16_32x1x1_mb16_gb512_4n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e_supple/4n8g/LibAI_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb512_4n8g_20220706_100943865207187/output.log">1207.65</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base_supple/4n8g/Megatron_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb512_4n8g_20220706_115955118787426.log">950.3</a>
                samples/s</td>
        </tr>
    </tbody>
</table>

<table class="docutils">
    <thead>
        <tr class="header">
            <th>GPT-2</th>
            <th>LiBai</th>
            <th>Megatron</th>
        </tr>
    </thead>
    <tbody>
        <tr class="odd">
            <td>nl24_fp16_1x1x1_mb6_gb6_1n1g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n1g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb6_gb6_1n1g_20220705_071259765473007/output.log">17.52</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n1g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb6_gb6_1n1g_20220615_075355864672227.log">15.5</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_4x1x1_mb4_gb16_1n4g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e_supple/1n4g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb16_1n4g_20220706_121838771888563/output.log">63.45</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base_supple/1n4g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb16_1n4g_20220706_121755031184092.log">53.3</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td>nl24_fp16_8x1x1_mb4_gb32_1n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb32_1n8g_20220705_091214203744961/output.log">125.64</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb32_1n8g_20220705_162733227027517.log">107.9</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_16x1x1_mb4_gb64_2n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb64_2n8g_20220705_153427485380612/output.log">215.35</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb64_2n8g_20220705_205510043191423.log">176.0</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td>nl24_fp16_32x1x1_mb4_gb128_4n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e_supple/4n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb128_4n8g_20220706_140324618820537/output.log">329.58</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base_supple/4n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb128_4n8g_20220706_123437709246728.log">296.6</a>
                samples/s</td>
        </tr>
    </tbody>
</table>

### Tensor Model Parallel 

<table class="docutils">
    <thead>
        <tr class="header">
            <th>BERT</th>
            <th>LiBai</th>
            <th>Megatron</th>
        </tr>
    </thead>
    <tbody>
        <tr class="odd">
            <td>nl24_fp16_1x1x1_ac_mb128_gb1024_1n1g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n1g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp1_mb128_gb1024_1n1g_20220705_071531848751549/output.log">35.74</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n1g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp1_mb128_gb1024_1n1g_20220615_131647218393872.log">33.6</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_1x4x1_ac_mb128_gb1024_1n4g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n4g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp4_pp1_mb128_gb1024_1n4g_20220705_091639328686421/output.log">87.12</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n4g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp4_pp1_mb128_gb1024_1n4g_20220705_122604083123137.log">86.6</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td>nl24_fp16_1x8x1_ac_mb128_gb1024_1n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp8_pp1_mb128_gb1024_1n8g_20220705_071502819874891/output.log">131.94</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp8_pp1_mb128_gb1024_1n8g_20220705_113839195864897.log">128.7</a>
                samples/s</td>
        </tr>
    </tbody>
</table>

<table class="docutils">
    <thead>
        <tr class="header">
            <th>GPT-2</th>
            <th>LiBai</th>
            <th>Megatron</th>
        </tr>
    </thead>
    <tbody>
        <tr class="odd">
            <td>nl24_fp16_1x1x1_mb6_gb6_1n1g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n1g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb6_gb6_1n1g_20220705_071259765473007/output.log">17.52</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n1g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb6_gb6_1n1g_20220615_075355864672227.log">15.5</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_1x4x1_mb6_gb6_1n4g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n4g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp4_pp1_mb6_gb6_1n4g_20220705_083540814077836/output.log">40.38</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n4g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp4_pp1_mb6_gb6_1n4g_20220705_161200662119880.log">38.0</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td>nl24_fp16_1x8x1_mb8_gb8_1n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp8_pp1_mb8_gb8_1n8g_20220705_071300514010057/output.log">60.53</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp8_pp1_mb8_gb8_1n8g_20220705_145234374022700.log">55.7</a>
                samples/s</td>
        </tr>
    </tbody>
</table>

### Pipeline Model Parallel

<table class="docutils">
    <thead>
        <tr class="header">
            <th>BERT</th>
            <th>LiBai</th>
            <th>Megatron</th>
        </tr>
    </thead>
    <tbody>
        <tr class="odd">
            <td>nl24_fp16_1x1x1_ac_mb128_gb1024_1n1g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n1g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp1_mb128_gb1024_1n1g_20220705_071531848751549/output.log">35.74</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n1g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp1_mb128_gb1024_1n1g_20220615_131647218393872.log">33.6</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_1x1x4_ac_mb128_gb1024_1n4g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n4g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb1024_1n4g_20220705_110658353978881/output.log">103.6</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n4g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb1024_1n4g_20220615_163155223131475.log">88.7</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td><strong>nl48</strong>_fp16_1x1x8_ac_mb64_gb1024_1n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_bert_nl48_nah16_hs1024_FP16_actrue_mp1_pp8_mb64_gb1024_1n8g_20220705_074452866672066/output.log">94.4</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_bert_nl48_nah16_hs1024_FP16_actrue_mp1_pp8_mb64_gb1024_1n8g_20220705_120956967492395.log">85.5</a>
                samples/s</td>
        </tr>
    </tbody>
</table>

<table class="docutils">
    <thead>
        <tr class="header">
            <th>GPT-2</th>
            <th>LiBai</th>
            <th>Megatron</th>
        </tr>
    </thead>
    <tbody>
        <tr class="odd">
            <td>nl24_fp16_1x1x1_ac_mb32_gb256_1n1g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n1g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp1_mb32_gb256_1n1g_20220705_071446147204953/output.log">14.43</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n1g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp1_mb32_gb256_1n1g_20220705_145945599193771.log">13.3</a>
                samples/</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_1x1x4_ac_mb32_gb256_1n4g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n4g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb256_1n4g_20220705_090306115011489/output.log">41.9</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n4g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb256_1n4g_20220615_111701194391665.log">33.2</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td><strong>nl48</strong>_fp16_1x1x8_ac_mb24_gb384_1n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_gpt2_nl48_nah16_hs1024_FP16_actrue_mp1_pp8_mb24_gb384_1n8g_20220705_075906245664894/output.log">37.4</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_gpt2_nl48_nah16_hs1024_FP16_actrue_mp1_pp8_mb24_gb384_1n8g_20220705_154144783493377.log">31.8</a>
                samples/s</td>
        </tr>
    </tbody>
</table>

### 2-D Parallel

#### Data Parallel + Tensor Model Parallel

<table class="docutils">
    <thead>
        <tr class="header">
            <th>BERT</th>
            <th>LiBai</th>
            <th>Megatron</th>
        </tr>
    </thead>
    <tbody>
        <tr class="odd">
            <td>nl24_fp16_2x2x1_ac_mb128_gb2048_1n4g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n4g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb128_gb2048_1n4g_20220705_140640645048573/output.log">88.47</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n4g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb128_gb2048_1n4g_20220615_171428527286012.log">86.6</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_4x2x1_ac_mb128_gb4096_1n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb128_gb4096_1n8g_20220705_121419365203845/output.log">175.94</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb128_gb4096_1n8g_20220615_162613310187064.log">172.0</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td>nl24_fp16_8x2x1_ac_mb128_gb8192_2n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb128_gb8192_2n8g_20220705_191030011908901/output.log">348.58</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb128_gb8192_2n8g_20220615_092121490236726.log">343.8</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_2x8x1_ac_mb128_gb2048_2n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp8_pp1_mb128_gb2048_2n8g_20220705_204305155951783/output.log">261.78</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp8_pp1_mb128_gb2048_2n8g_20220615_104722377958514.log">255.8</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td>nl24_fp16_4x4x1_ac_mb128_gb2048_2n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp4_pp1_mb128_gb2048_2n8g_20220705_184204966857940/output.log">338.97</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp4_pp1_mb128_gb2048_2n8g_20220705_203137819762324.log">337.3</a>
                samples/s</td>
        </tr>
    </tbody>
</table>

<table class="docutils">
    <thead>
        <tr class="header">
            <th>GPT-2</th>
            <th>LiBai</th>
            <th>Megatron</th>
        </tr>
    </thead>
    <tbody>
        <tr class="odd">
            <td>nl24_fp16_2x2x1_ac_mb32_gb512_1n4g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n4g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb32_gb512_1n4g_20220705_102345166928423/output.log">37.63</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n4g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb32_gb512_1n4g_20220615_114458702264816.log">36.9</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_4x2x1_ac_mb32_gb1024_1n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb32_gb1024_1n8g_20220705_103654387121991/output.log">74.35</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb32_gb1024_1n8g_20220615_102825468361561.log">73.2</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td>nl24_fp16_8x2x1_ac_mb32_gb2048_2n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb32_gb2048_2n8g_20220705_163225947465351/output.log">148.94</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb32_gb2048_2n8g_20220615_075410947484330.log">146.5</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_2x8x1_ac_mb32_gb512_2n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp8_pp1_mb32_gb512_2n8g_20220705_174941061081146/output.log">116.04</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp8_pp1_mb32_gb512_2n8g_20220616_090223352685185.log">109.1</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td>nl24_fp16_4x4x1_ac_mb32_gb512_2n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp4_pp1_mb32_gb512_2n8g_20220705_161315502270392/output.log">141.25</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp4_pp1_mb32_gb512_2n8g_20220615_084455786824917.log">138.1</a>
                samples/s</td>
        </tr>
    </tbody>
</table>

#### Data Parallel + Pipeline Model Parallel

<table class="docutils">
    <thead>
        <tr class="header">
            <th>BERT</th>
            <th>LiBai</th>
            <th>Megatron</th>
        </tr>
    </thead>
    <tbody>
        <tr class="odd">
            <td>nl24_fp16_2x1x4_ac_mb128_gb2048_1n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb2048_1n8g_20220705_135654422062875/output.log">207.31</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb2048_1n8g_20220705_140726038527715.log">175.0</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_4x1x4_ac_mb128_gb4096_2n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb4096_2n8g_20220705_211808588422098/output.log">406.24</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb4096_2n8g_20220615_121601428159750.log">342.9</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td>nl24_fp16_8x1x4_ac_mb128_gb8192_4n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e_supple/4n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb8192_4n8g_20220706_124739788495384/output.log">805.04</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base_supple/4n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb8192_4n8g_20220706_152441274628712.log">650.7</a>
                samples/s</td>
        </tr>
    </tbody>
</table>

<table class="docutils">
    <thead>
        <tr class="header">
            <th>GPT-2</th>
            <th>LiBai</th>
            <th>Megatron</th>
        </tr>
    </thead>
    <tbody>
        <tr class="odd">
            <td>nl24_fp16_2x1x4_ac_mb32_gb512_1n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb512_1n8g_20220705_120100257233978/output.log">83.12</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb512_1n8g_20220705_162859180952832.log">65.3</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_4x1x4_ac_mb32_gb1024_2n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb1024_2n8g_20220705_181145725094854/output.log">164.23</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb1024_2n8g_20220615_130009719082439.log">128.4</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td>nl24_fp16_8x1x4_ac_mb32_gb2048_4n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e_supple/4n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb2048_4n8g_20220706_145622217184041/output.log">322.42</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base_supple/4n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb2048_4n8g_20220706_142353564914037.log">247.3</a>
                samples/s</td>
        </tr>
    </tbody>
</table>

### 3-D Parallel

<table class="docutils">
    <thead>
        <tr class="header">
            <th>BERT</th>
            <th>LiBai</th>
            <th>Megatron</th>
        </tr>
    </thead>
    <tbody>
        <tr class="odd">
            <td>nl24_fp16_2x2x4_ac_mb128_gb2048_2n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb128_gb2048_2n8g_20220705_223156628574994/output.log">267.39</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb128_gb2048_2n8g_20220616_091946235804420.log">233.7</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_4x2x4_ac_mb192_gb6144_4n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/4n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb192_gb6144_4n8g_20220705_050226500268757/output.log">503.51</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/4n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb192_gb6144_4n8g_20220706_000244759822631.log">439.4</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td>nl24_fp16_2x4x4_ac_mb256_gb4096_4n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/4n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp4_pp4_mb256_gb4096_4n8g_20220705_062431065749653/output.log">405.75</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/4n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp4_pp4_mb256_gb4096_4n8g_20220616_023203818494929.log">338.7</a>
                samples/s</td>
        </tr>
    </tbody>
</table>

<table class="docutils">
    <thead>
        <tr class="header">
            <th>GPT-2</th>
            <th>LiBai</th>
            <th>Megatron</th>
        </tr>
    </thead>
    <tbody>
        <tr class="odd">
            <td>nl24_fp16_2x2x4_ac_mb32_gb1024_2n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb32_gb1024_2n8g_20220705_185756187637203/output.log">128.77</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb32_gb1024_2n8g_20220705_213345094190188.log">106.3</a>
                samples/s</td>
        </tr>
        <tr class="even">
            <td>nl24_fp16_4x2x4_ac_mb48_gb1536_4n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/4n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb48_gb1536_4n8g_20220705_035358751889185/output.log">209.32</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/4n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb48_gb1536_4n8g_20220706_005719759064651.log">179.5</a>
                samples/s</td>
        </tr>
        <tr class="odd">
            <td>nl24_fp16_2x4x4_ac_mb64_gb1024_4n8g</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/4n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp4_pp4_mb64_gb1024_4n8g_20220705_043108406236792/output.log">186.67</a>
                samples/s</td>
            <td><a
                    href="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/4n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp4_pp4_mb64_gb1024_4n8g_20220616_012941284271973.log">178.2</a>
                samples/s</td>
        </tr>
    </tbody>
</table>