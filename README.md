<!-- 配图 -->

<h2 align="center">LiBai: Toolbox for Large Scale Pretraining</h2>
<p align="center">
    <a href="https://pypi.org/project/LiBai/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/libai">
    </a>
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
</p>

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/Oneflow-Inc/libai/blob/main/README_zh-CN.md">简体中文</a>
    <p>
</h4>


## Introduction
LiBai is an open-source large-scale model pretraining toolbox with efficient parallelization techniques based on OneFlow. The main branch works with OneFlow 0.7.0.

<details open>
<summary> <b> Highlights </b> </summary>

- **Support a collection of parallel training components**

    LiBai provides multiple parallelisms, e.g. Data Parallelism, Tensor Parallelism, and Pipeline Parallelism. Besides, it's extensible for new parallelism.

- **Varied training techniques**

    Libai provides some out-of-the-box training techniques, e.g. Distributed Training, Mixed Precision Training, Activation Checkpointing, Recomputation, Gradient Accumulation, and Zero Redundancy Optimizer(ZeRO).

- **Support for both CV and NLP tasks**

    Predifined data process for both CV and NLP datasets, e.g. CIFAR, ImageNet, BERT Dataset.

- **Easy Usage**

    Components are designed to be modular in LiBai for better and easier usage as follows:
    - LazyConfig system for more flexible syntax and no predefined structures 
    - Friendly trainer and engine
    - Used as a library to support building research projects on top of it

- **High Efficience**

</details>

## Installation


## Getting Started


## Documentation
Please refer to [docs](https://libai.readthedocs.io/en/latest/index.html) for full API documentation and tutorials.

## ChangeLog

**Beta 0.1.0** was released in 15/2/2022:
- support deit training on imagenet
- ...

Please refer to [changelog.md](./changelog.md) for details and release history.

## Contributing

## Acknowledgement

## License

## Citation