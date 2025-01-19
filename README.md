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
    <a href="https://github.com/Oneflow-Inc/libai/issues">
        <img alt="Python Checks" src="https://github.com/Oneflow-Inc/libai/workflows/Python checks/badge.svg">
    </a>
    <a href="https://github.com/Oneflow-Inc/libai/issues">
        <img alt="Docs Release Status" src="https://github.com/Oneflow-Inc/libai/workflows/Document Release/badge.svg">
    </a>
</p>


## Introduction1

**English** | [简体中文](/README_zh-CN.md)

LiBai is a large-scale open-source model training toolbox based on OneFlow. The main branch works with OneFlow 0.7.0.

<details open>
<summary> <b> Highlights </b> </summary>

- **Support a collection of parallel training components**

    LiBai provides multiple parallelisms such as Data Parallelism, Tensor Parallelism, and Pipeline Parallelism. It's also extensible for other new parallelisms.

- **Varied training techniques**

    LiBai provides many out-of-the-box training techniques such as Distributed Training, Mixed Precision Training, Activation Checkpointing, Recomputation, Gradient Accumulation, and Zero Redundancy Optimizer(ZeRO).

- **Support for both CV and NLP tasks**

    LiBai has predefined data process for both CV and NLP datasets such as CIFAR, ImageNet, and BERT Dataset.

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

**Beta 0.3.0** was released in 03/11/2024, the general changes in **0.3.0** version are as follows:

**Features:**
- Support mock transformers, see [Mock transformers](https://github.com/Oneflow-Inc/libai/tree/main/projects/mock_transformers#readme)
- Support lm-evaluation-harness for model evaluation
- User Experience Optimization

**New Supported Models:**
- These models are natively supported by libai
<table class="docutils">
  <tbody>
    <tr>
      <th width="130"> Models </th>
      <th valign="bottom" align="center" width="140"> 2D(tp+pp) Inference</th>
      <th valign="bottom" align="center" width="140"> 3D Parallel Training </th>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/BLOOM"> <b> BLOOM </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/ChatGLM"> <b> ChatGLM </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">&#10004;</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/Couplets"> <b> Couplets </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">&#10004;</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/DALLE2"> <b> DALLE2 </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/Llama"> <b> Llama2 </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">&#10004;</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/MAE"> <b> MAE </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">&#10004;</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/Stable_Diffusion"> <b> Stable_Diffusion </b> </td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
  </tbody>
</table>

**New Mock Models:**
- These models are extended and implemented by libai through mocking transformers.
<table class="docutils">
  <tbody>
    <tr>
      <th width="130"> Models </th>
      <th valign="bottom" align="center" width="140">Tensor Parallel</th>
      <th valign="bottom" align="center" width="150">Pipeline Parallel</th>
    </tr>
    <tr>
      <td align="center"><a href="https://huggingface.co/docs/transformers/v4.26.1/en/model_doc/bloom#overview"> <b> BLOOM </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/openai/gpt-2/blob/master/model_card.md"> <b> GPT2 </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><a href="https://huggingface.co/docs/transformers/v4.28.0/en/model_doc/llama#overview"> <b> LLAMA </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><a href="https://huggingface.co/docs/transformers/main/en/model_doc/llama2"> <b> LLAMA2 </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><a href="https://huggingface.co/baichuan-inc/Baichuan-7B"> <b> Baichuan </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><a href="https://huggingface.co/docs/transformers/v4.26.1/en/model_doc/opt#overview"> <b> OPT </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
  </tbody>
</table>

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

## Join the WeChat group

![LiBai_Wechat_QRcode](./docs/source/tutorials/assets/LiBai_Wechat.png)