# Benchmarks

Here we provides our benchmark speed test results of LiBai's models compared with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) implementations. In LiBai V0.1.0, we only benchmark the speed tests under 8 GPUs and all of the experiments were conducted under the same settings for a fair comparison.

## Settings
### Environments
- Hardware: 8 NVIDIA V100s
- The commit of Megatron-LM for comparison: [commit](https://github.com/NVIDIA/Megatron-LM/commit/e156d2fea7fc5c98e645f7742eb86b643956d840)

### Model Hyper-parameters
- **BERT Model**
```python
num_layers = 24
num_attention_heads = 16
hidden_size = 1024
seq_length = 512
```
- **GPT-2 Model**
```python
num_layers = 24
num_attention_heads = 16
hidden_size = 1024
seq_length = 1024
```
- **T5 Model**
```python
num_layers = 12
num_attention_heads = 12
hidden_size = 768
seq_length = 512
```

## Main Results
Here we explain the evaluation indicators in the following tables:
- **Fp16**: Mixed Precision Training
- **Data**: Data Parallel Size
- **Model**: Tensor-Model Parallel Size
- **Pipeline**: Pipeline Parallel Size
- **MiB / Samples**: GPU Memory Cost & Throughput


**BERT Model**
<table class="docutils">
  <tbody>
    <tr>
      <th valign="bottom" align="center">Nodes / GPUs / Fp16</th>
      <th valign="bottom" align="center">Data / Model / Pipeline</th>
      <th valign="bottom" align="center">LiBai</th>
      <th valign="bottom" align="center">Megatron-LM</th>
    </tr>
    <tr>
      <td align="center"> 1 Nodes / 1 GPUs / Fp16 On </td>
      <td align="center"> Data=1 / Model=1 / Pipeline=1 </td>
      <td align="center"> 9904 MiB / 31.3 samples/s </td>
      <td align="center"> 9958 MiB / 29.7 samples/s </td>
    </tr>
    <tr>
      <td align="center"> 1 Nodes / 8 GPUs / Fp16 On </td>
      <td align="center"> Data=8 / Model=1 / Pipeline=1 </td>
      <td align="center"> 10456 MiB / 187.1 samples/s </td>
      <td align="center"> 10734 MiB / 192.2 samples/s </td>
    </tr>
    <tr>
      <td align="center"> 1 Nodes / 8 GPUs / Fp16 On </td>
      <td align="center"> Data=4 / Model=2 / Pipeline=1 </td>
      <td align="center"> 6854 MiB / 116.8 samples/s </td>
      <td align="center"> 6588 MiB / 109.2 samples/s </td>
    </tr>
    <tr>
      <td align="center"> 1 Nodes / 8 GPUs / Fp16 On </td>
      <td align="center"> Data=4 / Model=1 / Pipeline=2 </td>
      <td align="center"> 4420 MiB / 70.6 samples/s </td>
      <td align="center"> 5404 MiB / 89.2 samples/s </td>
    </tr>
    <tr>
      <td align="center"> 1 Nodes / 8 GPUs / Fp16 On </td>
      <td align="center"> Data=2 / Model=2 / Pipeline=2 </td>
      <td align="center"> 3226 MiB / 42.2 samples/s </td>
      <td align="center"> 3604 MiB / 45.5 samples/s </td>
    </tr>
    </tr>
  </tbody>
</table>

**GPT-2 Model**
<table class="docutils">
  <tbody>
    <tr>
      <th valign="bottom" align="center">Nodes / GPUs / Fp16</th>
      <th valign="bottom" align="center">Data / Model / Pipeline</th>
      <th valign="bottom" align="center">LiBai</th>
      <th valign="bottom" align="center">Megatron-LM</th>
    </tr>
    <tr>
      <td align="center"> 1 Nodes / 1 GPUs / Fp16 On </td>
      <td align="center"> Data=1 / Model=1 / Pipeline=1 </td>
      <td align="center"> 12594 MiB / 12.7 samples/s </td>
      <td align="center"> 12664 MiB / 12.5 samples/s </td>
    </tr>
    <tr>
      <td align="center"> 1 Nodes / 8 GPUs / Fp16 On </td>
      <td align="center"> Data=8 / Model=1 / Pipeline=1 </td>
      <td align="center"> 16010 MiB / 11.4 samples/s </td>
      <td align="center"> 14796 MiB / 14.1 samples/s </td>
    </tr>
    <tr>
      <td align="center"> 1 Nodes / 8 GPUs / Fp16 On </td>
      <td align="center"> Data=4 / Model=1 / Pipeline=2 </td>
      <td align="center"> 10636 MiB / 43.0 samples/s </td>
      <td align="center"> 10660 MiB / 55.7 samples/s </td>
    </tr>
    <tr>
      <td align="center"> 1 Nodes / 8 GPUs / Fp16 On </td>
      <td align="center"> Data=2 / Model=2 / Pipeline=2 </td>
      <td align="center"> 10110 MiB / 36.4 samples/s </td>
      <td align="center"> 10096 MiB / 48.1 samples/s </td>
    </tr>
    </tr>
  </tbody>
</table>

**T5 Model**
<table class="docutils">
  <tbody>
    <tr>
      <th valign="bottom" align="center">Nodes / GPUs / Fp16</th>
      <th valign="bottom" align="center">Data / Model / Pipeline</th>
      <th valign="bottom" align="center">LiBai</th>
      <th valign="bottom" align="center">Megatron-LM</th>
    </tr>
    <tr>
      <td align="center"> 1 Nodes / 1 GPUs / Fp16 On </td>
      <td align="center"> Data=1 / Model=1 / Pipeline=1 </td>
      <td align="center"> 13418 MiB / 76.6 samples/s </td>
      <td align="center"> 12308 MiB / 72.6 samples/s </td>
    </tr>
    <tr>
      <td align="center"> 1 Nodes / 8 GPUs / Fp16 On </td>
      <td align="center"> Data=8 / Model=1 / Pipeline=1 </td>
      <td align="center"> 13818 MiB / 458.9 samples/s </td>
      <td align="center"> 13084 MiB / 468.8 samples/s </td>
    </tr>
    <tr>
      <td align="center"> 1 Nodes / 8 GPUs / Fp16 On </td>
      <td align="center"> Data=4 / Model=2 / Pipeline=1 </td>
      <td align="center"> 8484 MiB / 256.9 samples/s </td>
      <td align="center"> 7888 MiB / 253.4 samples/s </td>
    </tr>
    <tr>
      <td align="center"> 1 Nodes / 8 GPUs / Fp16 On </td>
      <td align="center"> Data=4 / Model=1 / Pipeline=2 </td>
      <td align="center"> 4330 MiB / 136.2 samples/s </td>
      <td align="center"> 5340 MiB / 142.3 samples/s </td>
    </tr>
    <tr>
      <td align="center"> 1 Nodes / 8 GPUs / Fp16 On </td>
      <td align="center"> Data=2 / Model=2 / Pipeline=2 </td>
      <td align="center"> 4908 MiB / 45.1 samples/s </td>
      <td align="center"> 5010 MiB / 47.2 samples/s </td>
    </tr>
    </tr>
  </tbody>
</table>