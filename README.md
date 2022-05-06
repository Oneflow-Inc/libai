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

**Beta 0.1.0** was released in 22/03/2022, the main features and supported models in **0.1.0** version are as follows:

**Features:**
- Support Data Parallelism
- Support 1D Tensor Parallelism
- Support Pipeline Parallelism
- Unified distributed Layers for both single-GPU and multi-GPU training
- `LazyConfig` system for more flexible syntax and no predefined structures
- Easy-to-use trainer and engine
- Support both CV and NLP data processing
- Mixed Precision Training
- Activation Checkpointing
- Gradient Accumulation
- Gradient Clipping
- Zero Redundancy Optimizer (ZeRO)

**Supported Models:**
- Support 3D parallel [BERT](https://arxiv.org/abs/1810.04805) model
- Support 3D parallel [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) model
- Support 3D parallel [T5](https://arxiv.org/abs/1910.10683) model
- Support 3D parallel [Vision Transformer](https://arxiv.org/abs/2010.11929) model
- Support Data parallel [Swin Transformer](https://arxiv.org/abs/2103.14030) model
- Support finetune task in [QQP project](/projects/QQP/)
- Support text classification task in [text classification project](/projects/text_classification/)
- Support Pathways Language Model (PaLM) in [PaLM project](/projects/PaLM/)
- Support MoCo_v3 in [MOCOV3 project](/projects/MOCOV3/)
- (experimental) Support MAE in [MAE project](/projects/MAE/)

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
