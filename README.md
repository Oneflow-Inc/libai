<!-- 配图 -->

<h2 align="center">LiBai</h2>
<p align="center">
    <a href="https://libai.readthedocs.io/en/latest/index.html">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href="https://github.com/Oneflow-Inc/libai/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/Oneflow-Inc/libai.svg?color=blue">
    </a>
    <a href="https://github.com/Oneflow-Inc/libai/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/Oneflow-Inc/libai.svg">
    </a>
    <a href="https://github.com/Oneflow-Inc/libai/issues">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-pink.svg">
    </a>
    <a herf="https://github.com/Oneflow-Inc/libai/issues">
        <img alt="Python Checks" src="https://github.com/Oneflow-Inc/libai/workflows/Python checks/badge.svg">
    </a>
    <a herf="https://github.com/Oneflow-Inc/libai/issues">
        <img alt="Docs Release Status" src="https://github.com/Oneflow-Inc/libai/workflows/Document Release/badge.svg">
    </a>
</p>


## Introduction

**English** | [简体中文](/README_zh-CN.md)

LiBai is a large-scale open-source model training toolbox based on OneFlow. The main branch works with OneFlow 0.7.0.

<details open>
<summary> <b> Highlights </b> </summary>

- **Support a collection of parallel training components**

    LiBai provides multiple parallelisms such as Data Parallelism, Tensor Parallelism, and Pipeline Parallelism. It's also extensible for other new parallelisms.

- **Varied training techniques**

    LiBai provides many out-of-the-box training techniques such as Distributed Training, Mixed Precision Training, Activation Checkpointing, Recomputation, Gradient Accumulation, and Zero Redundancy Optimizer(ZeRO).

- **Support for both CV and NLP tasks**

    LiBai has predifined data process for both CV and NLP datasets such as CIFAR, ImageNet, and BERT Dataset.

- **Easy to use**

    LiBai's components are designed to be modular for easier usage as follows:
    - LazyConfig system for more flexible syntax and no predefined structures 
    - Friendly trainer and engine
    - Used as a library to support building research projects on it. See [projects/](/projects) for some projects that are built based on LiBai

- **High Efficiency**

</details>

## Installation

See [Installation instructions](https://libai.readthedocs.io/en/latest/tutorials/get_started/Installation.html).

## Getting Started

See [Quick Run](https://libai.readthedocs.io/en/latest/tutorials/get_started/quick_run.html) for the basic usage of LiBai.

## Documentation

See LiBai's [documentation](https://libai.readthedocs.io/en/latest/index.html) for full API documentation and tutorials.

## ChangeLog

**Beta 0.2.0** was released in 07/07/2022, the general changes in **0.2.0** version are as follows:

**Features:**
- Support evaluation enabled and set `eval_iter`
- Support customized sampler in `config.py`
- Support rdma for pipeline-model-parallel
- Support multi fused kernel 
   - fused_scale_mask_softmax_dropout
   - fused_scale_tril_softmax_mask_scale
   - fused_self_attention in branch `libai_bench`
- User Experience Optimization
- Optimization for training throughput, see [benchmark](https://libai.readthedocs.io/en/latest/tutorials/get_started/Benchmark.html) for more details

**Supported Models:**
- Support 3D parallel [Roberta](https://arxiv.org/abs/1907.11692) model
- Support 2D parallel (data parallel + tensor model parallel) [SimCSE](https://arxiv.org/abs/2104.08821) model
- Support Data parallel [MAE](https://arxiv.org/abs/2111.06377) model
- Support Data parallel [MOCOV3](https://arxiv.org/abs/2104.02057) model

See [changelog](./changelog.md) for details and release history.

## Contributing

We appreciate all contributions to improve LiBai. See [CONTRIBUTING](./CONTRIBUTING.md) for the contributing guideline.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you find this project useful for your research, consider cite:

```BibTeX
@misc{of2021libai,
  author =       {Xingyu Liao and Peng Cheng and Tianhe Ren and Depeng Liang and
                  Kai Dang and Yi Wang and Xiaoyu Xu},
  title =        {LiBai},
  howpublished = {\url{https://github.com/Oneflow-Inc/libai}},
  year =         {2021}
}
```
