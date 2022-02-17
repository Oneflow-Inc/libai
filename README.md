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

LiBai is an open-source large-scale model training toolbox based on OneFlow. The main branch works with OneFlow 0.7.0.

<details open>
<summary> <b> Highlights </b> </summary>

- **Support a collection of parallel training components**

    LiBai provides multiple parallelisms, e.g. Data Parallelism, Tensor Parallelism, and Pipeline Parallelism. Besides, it's extensible for new parallelism.

- **Varied training techniques**

    LiBai provides some out-of-the-box training techniques, e.g. Distributed Training, Mixed Precision Training, Activation Checkpointing, Recomputation, Gradient Accumulation, and Zero Redundancy Optimizer(ZeRO).

- **Support for both CV and NLP tasks**

    Predifined data process for both CV and NLP datasets, e.g. CIFAR, ImageNet, BERT Dataset.

- **Easy Usage**

    Components are designed to be modular in LiBai for better and easier usage as follows:
    - LazyConfig system for more flexible syntax and no predefined structures 
    - Friendly trainer and engine
    - Used as a library to support building research projects on top of it. Please see [projects/](/projects) for some projects that are built on top of LiBai.

- **High Efficience**

</details>

## Installation
Please refer to [Installation.md](https://libai.readthedocs.io/en/latest/tutorials/Installation.html) for installation

## Getting Started
Please refer to [Getting Started.md](https://libai.readthedocs.io/en/latest/tutorials/Getting%20Started.html) for basic usage of LiBai

## Documentation
Please refer to [docs](https://libai.readthedocs.io/en/latest/index.html) for full API documentation and tutorials

## ChangeLog

**Beta 0.1.0** was released in 15/2/2022:
- Support 3D parallelism [BERT](https://arxiv.org/abs/1810.04805) models for pretraining.
- Support 2D Parallelism [ViT](https://arxiv.org/abs/2010.11929) models for Image Classification.

Please refer to [changelog.md](./changelog.md) for details and release history.

## Contributing
We appreciate all contributions to improve LiBai. Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for the contributing guideline.

## License
This project is released under the [Apache 2.0 license](LICENSE).

## Citation
If you find this project useful for your research, please consider cite:
```BibTeX
@misc{of2021libai,
  author =       {Xingyu Liao and Peng Cheng and Tianhe Ren and Depeng Liang and
                  Kai Dang and Yi Wang and Xiaoyu Xu},
  title =        {LiBai},
  howpublished = {\url{https://github.com/Oneflow-Inc/libai}},
  year =         {2021}
}
```