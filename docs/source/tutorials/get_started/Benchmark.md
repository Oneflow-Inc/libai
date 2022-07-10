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
- **dxmxp**:
  - d: data-parallel-size
  - m: tensor-model-parallel-size
  - p: pipeline-model-parallel-size
- **1n1g**: 1 node, 1 gpu
- **2n8g**: 2 nodes, 8 gpus per node, 16 gpus in total
- **4n8g**: 4 nodes, 8 gpus per node, 32 gpus in total
- `grad_acc_num_step = global_batch_size / (micro_batch_size * data_parallel_size)`
- **samples/s**: throughput


### Data Parallel

| BERT                             | LiBai                                                        | Megatron                                                     |
| -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| nl24_fp16_1x1x1_mb24_gb24_1n1g   | [46.91](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n1g/LibAI_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb24_gb24_1n1g_20220705_071307389288504/output.log) samples/s | [42.6](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n1g/Megatron_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb24_gb24_1n1g_20220615_130039677349789.log) samples/s |
| nl24_fp16_8x1x1_mb16_gb64_1n4g   | [176.88](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e_supple/1n4g/LibAI_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb64_1n4g_20220706_103618805733678/output.log) samples/s | [154.7](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base_supple/1n4g/Megatron_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb64_1n4g_20220706_121753217673018.log) samples/s |
| nl24_fp16_8x1x1_mb16_gb128_1n8g  | [351.57](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb128_1n8g_20220705_101124804210475/output.log) samples/s | [309.2](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb128_1n8g_20220705_140535074517604.log) samples/s |
| nl24_fp16_16x1x1_mb16_gb256_2n8g | [675.87](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb256_2n8g_20220705_172421459267607/output.log) samples/s | [534.7](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb256_2n8g_20220705_193107517518321.log) samples/s |
| nl24_fp16_32x1x1_mb16_gb512_4n8g | [1207.65](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e_supple/4n8g/LibAI_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb512_4n8g_20220706_100943865207187/output.log) samples/s | [950.3](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base_supple/4n8g/Megatron_bert_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb16_gb512_4n8g_20220706_115955118787426.log) samples/s |

| GPT-2                           | LiBai                                                        | Megatron                                                     |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| nl24_fp16_1x1x1_mb6_gb6_1n1g    | [17.52](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n1g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb6_gb6_1n1g_20220705_071259765473007/output.log) samples/s | [15.5](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n1g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb6_gb6_1n1g_20220615_075355864672227.log) samples/s |
| nl24_fp16_4x1x1_mb4_gb16_1n4g   | [63.45](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e_supple/1n4g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb16_1n4g_20220706_121838771888563/output.log) samples/s | [53.3](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base_supple/1n4g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb16_1n4g_20220706_121755031184092.log) samples/s |
| nl24_fp16_8x1x1_mb4_gb32_1n8g   | [125.64](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb32_1n8g_20220705_091214203744961/output.log) samples/s | [107.9](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb32_1n8g_20220705_162733227027517.log) samples/s |
| nl24_fp16_16x1x1_mb4_gb64_2n8g  | [215.35](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb64_2n8g_20220705_153427485380612/output.log) samples/s | [176.0](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb64_2n8g_20220705_205510043191423.log) samples/s |
| nl24_fp16_32x1x1_mb4_gb128_4n8g | [329.58](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e_supple/4n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb128_4n8g_20220706_140324618820537/output.log) samples/s | [296.6](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base_supple/4n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb4_gb128_4n8g_20220706_123437709246728.log) samples/s |

### Tensor Model Parallel 

| BERT                                 | LiBai                                                        | Megatron                                                     |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| nl24_fp16_1x1x1_ac_mb128_gb1024_1n1g | [35.74](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n1g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp1_mb128_gb1024_1n1g_20220705_071531848751549/output.log) samples/s | [33.6](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n1g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp1_mb128_gb1024_1n1g_20220615_131647218393872.log) samples/s |
| nl24_fp16_1x4x1_ac_mb128_gb1024_1n4g | [87.12](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n4g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp4_pp1_mb128_gb1024_1n4g_20220705_091639328686421/output.log) samples/s | [86.6](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n4g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp4_pp1_mb128_gb1024_1n4g_20220705_122604083123137.log) samples/s |
| nl24_fp16_1x8x1_ac_mb128_gb1024_1n8g | [131.94](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp8_pp1_mb128_gb1024_1n8g_20220705_071502819874891/output.log) samples/s | [128.7](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp8_pp1_mb128_gb1024_1n8g_20220705_113839195864897.log) samples/s |

| GPT-2                        | LiBai                                                        | Megatron                                                     |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| nl24_fp16_1x1x1_mb6_gb6_1n1g | [17.52](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n1g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb6_gb6_1n1g_20220705_071259765473007/output.log) samples/s | [15.5](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n1g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp1_pp1_mb6_gb6_1n1g_20220615_075355864672227.log) samples/s |
| nl24_fp16_1x4x1_mb6_gb6_1n4g | [40.38](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n4g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp4_pp1_mb6_gb6_1n4g_20220705_083540814077836/output.log) samples/s | [38.0](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n4g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp4_pp1_mb6_gb6_1n4g_20220705_161200662119880.log) samples/s |
| nl24_fp16_1x8x1_mb8_gb8_1n8g | [60.53](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp8_pp1_mb8_gb8_1n8g_20220705_071300514010057/output.log) samples/s | [55.7](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_acfalse_mp8_pp1_mb8_gb8_1n8g_20220705_145234374022700.log) samples/s |

### Pipeline Model Parallel

| BERT                                    | LiBai                                                        | Megatron                                                     |
| --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| nl24_fp16_1x1x1_ac_mb128_gb1024_1n1g    | [35.74](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n1g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp1_mb128_gb1024_1n1g_20220705_071531848751549/output.log) samples/s | [33.6](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n1g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp1_mb128_gb1024_1n1g_20220615_131647218393872.log) samples/s |
| nl24_fp16_1x1x4_ac_mb128_gb1024_1n4g    | [103.6](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n4g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb1024_1n4g_20220705_110658353978881/output.log) samples/s | [88.7](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n4g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb1024_1n4g_20220615_163155223131475.log) samples/s |
| **nl48**_fp16_1x1x8_ac_mb64_gb1024_1n8g | [94.4](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_bert_nl48_nah16_hs1024_FP16_actrue_mp1_pp8_mb64_gb1024_1n8g_20220705_074452866672066/output.log) samples/s | [85.5](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_bert_nl48_nah16_hs1024_FP16_actrue_mp1_pp8_mb64_gb1024_1n8g_20220705_120956967492395.log) samples/s |

| GPT-2                                  | LiBai                                                        | Megatron                                                     |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| nl24_fp16_1x1x1_ac_mb32_gb256_1n1g     | [14.43](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n1g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp1_mb32_gb256_1n1g_20220705_071446147204953/output.log) samples/s | [13.3](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n1g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp1_mb32_gb256_1n1g_20220705_145945599193771.log) samples/ |
| nl24_fp16_1x1x4_ac_mb32_gb256_1n4g     | [41.9](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n4g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb256_1n4g_20220705_090306115011489/output.log) samples/s | [33.2](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n4g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb256_1n4g_20220615_111701194391665.log) samples/s |
| **nl48**_fp16_1x1x8_ac_mb24_gb384_1n8g | [37.4](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_gpt2_nl48_nah16_hs1024_FP16_actrue_mp1_pp8_mb24_gb384_1n8g_20220705_075906245664894/output.log) samples/s | [31.8](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_gpt2_nl48_nah16_hs1024_FP16_actrue_mp1_pp8_mb24_gb384_1n8g_20220705_154144783493377.log) samples/s |

### 2-D Parallel

#### Data Parallel + Tensor Model Parallel

| BERT                                 | LiBai                                                        | Megatron                                                     |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| nl24_fp16_2x2x1_ac_mb128_gb2048_1n4g | [88.47](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n4g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb128_gb2048_1n4g_20220705_140640645048573/output.log) samples/s | [86.6](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n4g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb128_gb2048_1n4g_20220615_171428527286012.log) samples/s |
| nl24_fp16_4x2x1_ac_mb128_gb4096_1n8g | [175.94](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb128_gb4096_1n8g_20220705_121419365203845/output.log) samples/s | [172.0](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb128_gb4096_1n8g_20220615_162613310187064.log) samples/s |
| nl24_fp16_8x2x1_ac_mb128_gb8192_2n8g | [348.58](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb128_gb8192_2n8g_20220705_191030011908901/output.log) samples/s | [343.8](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb128_gb8192_2n8g_20220615_092121490236726.log) samples/s |
| nl24_fp16_2x8x1_ac_mb128_gb2048_2n8g | [261.78](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp8_pp1_mb128_gb2048_2n8g_20220705_204305155951783/output.log) samples/s | [255.8](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp8_pp1_mb128_gb2048_2n8g_20220615_104722377958514.log) samples/s |
| nl24_fp16_4x4x1_ac_mb128_gb2048_2n8g | [338.97](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp4_pp1_mb128_gb2048_2n8g_20220705_184204966857940/output.log) samples/s | [337.3](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp4_pp1_mb128_gb2048_2n8g_20220705_203137819762324.log) samples/s |

| GPT-2                               | LiBai                                                        | Megatron                                                     |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| nl24_fp16_2x2x1_ac_mb32_gb512_1n4g  | [37.63](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n4g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb32_gb512_1n4g_20220705_102345166928423/output.log) samples/s | [36.9](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n4g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb32_gb512_1n4g_20220615_114458702264816.log) samples/s |
| nl24_fp16_4x2x1_ac_mb32_gb1024_1n8g | [74.35](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb32_gb1024_1n8g_20220705_103654387121991/output.log) samples/s | [73.2](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb32_gb1024_1n8g_20220615_102825468361561.log) samples/s |
| nl24_fp16_8x2x1_ac_mb32_gb2048_2n8g | [148.94](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb32_gb2048_2n8g_20220705_163225947465351/output.log) samples/s | [146.5](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp1_mb32_gb2048_2n8g_20220615_075410947484330.log) samples/s |
| nl24_fp16_2x8x1_ac_mb32_gb512_2n8g  | [116.04](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp8_pp1_mb32_gb512_2n8g_20220705_174941061081146/output.log) samples/s | [109.1](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp8_pp1_mb32_gb512_2n8g_20220616_090223352685185.log) samples/s |
| nl24_fp16_4x4x1_ac_mb32_gb512_2n8g  | [141.25](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp4_pp1_mb32_gb512_2n8g_20220705_161315502270392/output.log) samples/s | [138.1](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp4_pp1_mb32_gb512_2n8g_20220615_084455786824917.log) samples/s |

#### Data Parallel + Pipeline Model Parallel

| BERT                                 | LiBai                                                        | Megatron                                                     |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| nl24_fp16_2x1x4_ac_mb128_gb2048_1n8g | [207.31](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb2048_1n8g_20220705_135654422062875/output.log) samples/s | [175.0](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb2048_1n8g_20220705_140726038527715.log) samples/s |
| nl24_fp16_4x1x4_ac_mb128_gb4096_2n8g | [406.24](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb4096_2n8g_20220705_211808588422098/output.log) samples/s | [342.9](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb4096_2n8g_20220615_121601428159750.log) samples/s |
| nl24_fp16_8x1x4_ac_mb128_gb8192_4n8g | [805.04](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e_supple/4n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb8192_4n8g_20220706_124739788495384/output.log) samples/s | [650.7](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base_supple/4n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb128_gb8192_4n8g_20220706_152441274628712.log) samples/s |

| GPT-2                               | LiBai                                                        | Megatron                                                     |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| nl24_fp16_2x1x4_ac_mb32_gb512_1n8g  | [83.12](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/1n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb512_1n8g_20220705_120100257233978/output.log) samples/s | [65.3](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/1n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb512_1n8g_20220705_162859180952832.log) samples/s |
| nl24_fp16_4x1x4_ac_mb32_gb1024_2n8g | [164.23](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb1024_2n8g_20220705_181145725094854/output.log) samples/s | [128.4](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb1024_2n8g_20220615_130009719082439.log) samples/s |
| nl24_fp16_8x1x4_ac_mb32_gb2048_4n8g | [322.42](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e_supple/4n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb2048_4n8g_20220706_145622217184041/output.log) samples/s | [247.3](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base_supple/4n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp1_pp4_mb32_gb2048_4n8g_20220706_142353564914037.log) samples/s |

### 3-D Parallel

| BERT                                 | LiBai                                                        | Megatron                                                     |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| nl24_fp16_2x2x4_ac_mb128_gb2048_2n8g | [267.39](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb128_gb2048_2n8g_20220705_223156628574994/output.log) samples/s | [233.7](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb128_gb2048_2n8g_20220616_091946235804420.log) samples/s |
| nl24_fp16_4x2x4_ac_mb192_gb6144_4n8g | [503.51](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/4n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb192_gb6144_4n8g_20220705_050226500268757/output.log) samples/s | [439.4](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/4n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb192_gb6144_4n8g_20220706_000244759822631.log) samples/s |
| nl24_fp16_2x4x4_ac_mb256_gb4096_4n8g | [405.75](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/4n8g/LibAI_bert_nl24_nah16_hs1024_FP16_actrue_mp4_pp4_mb256_gb4096_4n8g_20220705_062431065749653/output.log) samples/s | [338.7](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/4n8g/Megatron_bert_nl24_nah16_hs1024_FP16_actrue_mp4_pp4_mb256_gb4096_4n8g_20220616_023203818494929.log) samples/s |

| GPT-2                               | LiBai                                                        | Megatron                                                     |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| nl24_fp16_2x2x4_ac_mb32_gb1024_2n8g | [128.77](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/2n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb32_gb1024_2n8g_20220705_185756187637203/output.log) samples/s | [106.3](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/2n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb32_gb1024_2n8g_20220705_213345094190188.log) samples/s |
| nl24_fp16_4x2x4_ac_mb48_gb1536_4n8g | [209.32](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/4n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb48_gb1536_4n8g_20220705_035358751889185/output.log) samples/s | [179.5](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/4n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp2_pp4_mb48_gb1536_4n8g_20220706_005719759064651.log) samples/s |
| nl24_fp16_2x4x4_ac_mb64_gb1024_4n8g | [186.67](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/55b822e/4n8g/LibAI_gpt2_nl24_nah16_hs1024_FP16_actrue_mp4_pp4_mb64_gb1024_4n8g_20220705_043108406236792/output.log) samples/s | [178.2](https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base/4n8g/Megatron_gpt2_nl24_nah16_hs1024_FP16_actrue_mp4_pp4_mb64_gb1024_4n8g_20220616_012941284271973.log) samples/s |