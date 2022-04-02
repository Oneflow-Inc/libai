## Changelog

### Beta 0.1.0 (22/03/2022)

**New Features:**
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
- Support 3D parallel [Vision Transformer](https://arxiv.org/abs/2010.11929)
- Support Data parallel [Swin Transformer](https://arxiv.org/abs/2103.14030) model
- Support finetune task in [QQP project](/projects/QQP/)
- Support text classification task in [text classification project](/projects/text_classification/)
