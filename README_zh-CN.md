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

## 简介

[English](/README.md) | **简体中文**

LiBai是一个基于OneFlow的大规模模型训练开源工具箱，主分支代码目前支持OneFlow 0.7.0以上的版本。

<details open>
<summary> <b> 主要特性 </b> </summary>

- **支持丰富的并行训练配置**

    LiBai支持了丰富的并行训练配置，包括数据并行、模型并行、流水并行等并行方式。可拓展性好，易于拓展到更丰富的并行训练模式上。

- **多样化的训练技巧**

    LiBai提供了丰富的开箱即用的训练技巧，包括但不限于分布式训练、混合精度训练、后向重计算、Zero Redundancy Optimizer(ZeRO)等训练方式。

- **同时支持视觉与自然语言处理任务**

    LiBai中内置了CV与NLP相关的数据集处理流程，包括CIFAR、ImageNet、BERT Dataset等数据集。

- **简单易用，便于上手**

    LiBai的模块化设计可以让用户更为方便地将LiBai拓展到自己的项目上:
    - 配置系统采用LazyConfig方式，使得配置系统更加灵活且易于拓展
    - 采用Trainer与Hook结合的方式，方便用户使用和拓展训练中需要的组件
    - 用户可以在安装好LiBai的基础上灵活地开发自己的任务，而非强依赖于LiBai中的所有组件。可以查看[基于LiBai的项目](/projects)了解更多细节

- **速度快，性能高**

</details>

## 安装
请参考[LiBai安装文档](https://libai.readthedocs.io/en/latest/tutorials/get_started/Installation.html)进行安装。

## 快速入门
请参考[快速入门文档](https://libai.readthedocs.io/en/latest/tutorials/get_started/quick_run.html)了解和学习LiBai的基本使用，后续我们将提供丰富的教程与完整的使用指南。

## 使用文档
请参考[LiBai使用文档](https://libai.readthedocs.io/en/latest/index.html)了解LiBai中相关接口的使用。

## 更新日志

最新的 **Beta 0.1.0** 版本已经在 2022.03.22 发布, 目前 **0.1.0** 版本主要支持以下特性及模型:

**相关特性:**
- 数据并行 (Data Parallelism)
- 1维张量并行 (1D Tensor Parallelism)
- 流水并行 (Pipeline Parallelism)
- 单卡和多卡统一的分布式神经网络层 (Unified Distributed Layers)
- 可灵活扩展新的并行方式 (Extensible for new parallelism)
- 混合精度训练 (Mixed Precision Training)
- 后向重计算 (Activation Checkpointing)
- 梯度累加 (Gradient Accumulation)
- 梯度裁剪 (Gradient Clip)
- 零冗余优化器 (ZeRO)
- 更灵活的 `LazyConfig` 配置系统
- 易于使用的 `Trainer` 和 `Evaluator`
- 同时支持图像和文本的数据预处理

**支持的模型:**
- 支持3D并行的[BERT](https://arxiv.org/abs/1810.04805)模型
- 支持3D并行的[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)模型
- 支持3D并行的[T5](https://arxiv.org/abs/1910.10683)模型
- 支持3D并行的[Vision Transformer](https://arxiv.org/abs/2010.11929)模型
- 支持数据并行的[Swin Transformer](https://arxiv.org/abs/2103.14030)模型
- 在[QQP project](/projects/QQP/)中支持相关的finetune任务
- 在[text classification project](/projects/text_classification/)中支持文本分类任务

历史版本的发布与更新细节请参考[更新日志](./changelog.md)。

## 参与贡献

我们欢迎任何有助于提升LiBai的贡献. 请参考[贡献指南](./CONTRIBUTING.md)来了解如何参与贡献。

## 许可证

该项目开源自[Apache 2.0 license](LICENSE)。

## Citation

如果LiBai对于你的研究项目有帮助的话，请参考如下的BibTeX引用LiBai：

```BibTeX
@misc{of2021libai,
  author =       {Xingyu Liao and Peng Cheng and Tianhe Ren and Depeng Liang and
                  Kai Dang and Yi Wang and Xiaoyu Xu},
  title =        {LiBai},
  howpublished = {\url{https://github.com/Oneflow-Inc/libai}},
  year =         {2021}
}
```

