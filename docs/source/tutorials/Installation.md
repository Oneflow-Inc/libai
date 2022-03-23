# Installation
LiBai provides two installation modes to meet different needs:
- [Use as a librar](#use-libai-as-a-python-library) to call LiBai's APIs or import LiBai's module in your own project.
- [Build from source](#build-libai-from-source) to develop your own project based on LiBai's framework.

## Use LiBai as a Python Library

```bash
pip install libai
```

## Build LiBai from Source

- Clone this repo:

```bash
git clone https://github.com/Oneflow-Inc/libai.git
cd libai
```
- Create a conda virtual environment and activate it:

```bash
conda create -n libai python=3.7 -y
conda activate libai
```

- Install the stable release of OneFlow with `CUDA` support refer to [OneFlow installation guide](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package).
- Install `pybind11`

```bash
pip install pybind11
```

- For an editable installation of LiBai:

```bash
pip install -e .
```


