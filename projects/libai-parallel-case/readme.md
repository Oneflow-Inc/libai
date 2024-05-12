### libai-parallel-case

- launch

```bash
bash tools/train.sh tools/train_net.py projects/libai-parallel-case/configs/gpt2_pretrain_data_parallel.py 8

bash tools/train.sh tools/train_net.py projects/libai-parallel-case/configs/gpt2_pretrain_tensor_parallel.py 8

bash tools/train.sh tools/train_net.py projects/libai-parallel-case/configs/gpt2_pretrain_pipeline_parallel.py 8

bash tools/train.sh tools/train_net.py projects/libai-parallel-case/configs/gpt2_pretrain_singlecard.py 1

bash tools/train.sh tools/train_net.py projects/libai-parallel-case/configs/gpt2_pretrain_auto_parallel.py 8

bash tools/train.sh tools/train_net.py projects/libai-parallel-case/configs/gpt2_pretrain_singlecard_oom.py 1

```

- total params

```python
wte: 50257 * 1600 = 80411200
wpe: 1024 * 1600 = 1638400
ln_f: 1600
lm_head: 1600 * 50257 = 80411200
h: {
    ln_1: 1600
    attn: {
        c_attn: 3 * 1600 * 1600 = 7680000
        c_proj: 1600 * 1600 = 2560000
    }
    ln_2: 1600
    mlp: {
        c_fc: 4 * 1600 * 1600 = 10240000
        c_proj: 1600 * 4 * 1600 = 10240000
    }
} = 30723200

80411200 + 1638400 + 1600 + 80411200 + 30723200 * 48 = 1637176000
```


- download test datasets

```
[BIN_DATA_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.bin)

[IDX_DATA_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.idx)
```

- 查看软件、硬件信息
1、查看cpu信息：lscpu

```
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         46 bits physical, 57 bits virtual
  Byte Order:            Little Endian
CPU(s):                  56
  On-line CPU(s) list:   0-55
Vendor ID:               GenuineIntel
  Model name:            Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz
    CPU family:          6
    Model:               106
    Thread(s) per core:  1
    Core(s) per socket:  28
    Socket(s):           2
    Stepping:            6
    CPU max MHz:         3500.0000
    CPU min MHz:         800.0000
    BogoMIPS:            5200.00
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr
                          sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_go
                         od nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm
                         2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes
                          xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 invpcid_single intel_ppin
                          ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc
                         _adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma cl
                         flushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc c
                         qm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect wbnoinvd dtherm ida arat pln pts avx51
                         2vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopc
                         ntdq la57 rdpid fsrm md_clear pconfig flush_l1d arch_capabilities
Virtualization features: 
  Virtualization:        VT-x
Caches (sum of all):     
  L1d:                   2.6 MiB (56 instances)
  L1i:                   1.8 MiB (56 instances)
  L2:                    70 MiB (56 instances)
  L3:                    84 MiB (2 instances)
NUMA:                    
  NUMA node(s):          2
  NUMA node0 CPU(s):     0-27
  NUMA node1 CPU(s):     28-55
Vulnerabilities:         
  Gather data sampling:  Mitigation; Microcode
  Itlb multihit:         Not affected
  L1tf:                  Not affected
  Mds:                   Not affected
  Meltdown:              Not affected
  Mmio stale data:       Mitigation; Clear CPU buffers; SMT disabled
  Retbleed:              Not affected
  Spec rstack overflow:  Not affected
  Spec store bypass:     Mitigation; Speculative Store Bypass disabled via prctl and seccomp
  Spectre v1:            Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:            Mitigation; Enhanced IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
  Srbds:                 Not affected
  Tsx async abort:       Not affected
```

- 查看GPU信息：gpustat

```
ubuntu21                  Fri May 10 23:10:19 2024  545.23.06
[0] NVIDIA A100-PCIE-40GB | 32°C,   0 % |     7 / 40960 MB |
[1] NVIDIA A100-PCIE-40GB | 34°C,   0 % |     7 / 40960 MB |
[2] NVIDIA A100-PCIE-40GB | 33°C,   0 % |     7 / 40960 MB |
[3] NVIDIA A100-PCIE-40GB | 33°C,   0 % |     7 / 40960 MB |
[4] NVIDIA A100-PCIE-40GB | 32°C,   0 % |     7 / 40960 MB |
[5] NVIDIA A100-PCIE-40GB | 32°C,   0 % |     7 / 40960 MB |
[6] NVIDIA A100-PCIE-40GB | 32°C,   0 % |     7 / 40960 MB |
[7] NVIDIA A100-PCIE-40GB | 32°C,   0 % |     7 / 40960 MB |
```

- 查看cuda版本：nvcc --version

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Sep__8_19:17:24_PDT_2023
Cuda compilation tools, release 12.3, V12.3.52
Build cuda_12.3.r12.3/compiler.33281558_0
```

- 查看cuda驱动版本：cat /proc/driver/nvidia/version

```
NVRM version: NVIDIA UNIX x86_64 Kernel Module  545.23.06  Sun Oct 15 17:43:11 UTC 2023
GCC version:  gcc version 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04)
```
