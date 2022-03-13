<!-- 配图 -->
 LiBai的模块化设计可以让用户更为方便地将LiBai拓展到自己的项目上:
    - 配置系统采用LazyConfig方式，使得配置系统更加灵活且易于拓展
    - 采用Trainer与Hook结合的方式，方便用户使用和拓展训练中需要的组件
    - 用户可以在安装好LiBai的基础上灵活地开发自己的任务，而非强依赖于LiBai中的所有组件。可以查看[基于LiBai的项目](/projects)了解更多细节

- **速度快，性能高**

</details>

## 安装
请参考[LiBai安装文档](https://libai.readthedocs.io/en/latest/tutorials/Installation.html)进行安装。

## 快速入门
请参考[快速入门文档](https://libai.readthedocs.io/en/latest/tutorials/Getting%20Started.html)了解和学习LiBai的基本使用，后续我们将提供丰富的教程与完整的使用指南。

## 使用文档
请参考[LiBai使用文档](https://libai.readthedocs.io/en/latest/index.html)了解LiBai中相关接口的使用。

## 更新日志

最新的**Beta 0.1.0**版本已经在 2022.02.15 发布：
- 支持2D并行ViT模型在ImageNet上的完整训练
- 支持3D并行的BERT模型预训练

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
