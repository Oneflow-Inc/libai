<!-- 配图 -->

<h2 align="center">LiBai</h2>
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

See [Installation instructions](https://libai.readthedocs.io/en/latest/tutorials/Installation.html).

## Getting Started

See [Getting Started](https://libai.readthedocs.io/en/latest/tutorials/Getting%20Started.html) for the basic usage of LiBai.

## Documentation

See LiBai's [documentation](https://libai.readthedocs.io/en/latest/index.html) for full API documentation and tutorials.

## ChangeLog

**Beta 0.1.0** was released in 15/2/2022:
- Support 3D parallelism [BERT](https://arxiv.org/abs/1810.04805) models for pretraining.
- Support 2D parallelism [ViT](https://arxiv.org/abs/2010.11929) models for image classification.

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
