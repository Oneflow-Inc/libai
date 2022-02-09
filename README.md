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
        <a href="">简体中文</a>
    <p>
</h4>


## Introduction
Libai is an open source large pretraining toolbox based on OneFlow. The main branch works with OneFlow 0.7.0.

<details open>
<summary> <b> Highlights </b> </summary>

- **Support of multiple parallel strategy**

    The toolbox supports multiple parallelism, e.g. Data Parallelism, Tensor Parallelism, and Pipeline Parallelism

- **Varied training techniques**

    Libai directly support the state-of-art training techniques, e.g. Distributed Training, Mixed Precision training, Activation Checkpoint, and Zero Redundance Optimizer

- **Support for both CV and NLP tasks**

    Predifined data process for both CV and NLP datasets, e.g. CIFAR, ImageNet.

- **Easy Usage**

    LazyConfig system for more flexible syntax and no predefined structures and easy-to-use trainer engine.

- **High Efficience**

</details>

<!-- <details open>
<summary> <b> Libai supported projects </b> </summary>

- Masked word completion with [BERT](https://arxiv.org/abs/1810.04805)

- Image classification with [ViT](https://arxiv.org/abs/2010.11929)

</details> -->


## Documentation
Please refer to [docs]() for full API documentation and tutorials.

## License
This project is released under the [Apache 2.0 license](LICENSE).


## ChangeLog

**Beta 0.1.0** was released in 15/2/2022:
- support deit training on imagenet
- ...

Please refer to [changelog.md]() for details and release history.

## Supported Projects
- [x] 3D Parallelism [BERT](https://arxiv.org/abs/1810.04805) models for pretraining.
- [x] 2D Parallelism [ViT](https://arxiv.org/abs/2010.11929) models for Image Classification.

## Installation

## Getting Started

## Contributing

## Acknowledgement