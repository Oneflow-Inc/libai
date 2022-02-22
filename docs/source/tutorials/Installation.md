# Installation
LiBai provides two installation modes to meet different needs:
- [Use as a library](#use-libai-as-a-python-library) to call LiBai's APIs or import LiBai's Module in your own project.
- [Build from source](#build-libai-from-source) to develop your own project based on LiBai's framework.

## Use LiBai as a Python Library

```bash
pip install libai
```

## Build LiBai from Source

1. Clone this repo:

  ```bash
  git clone https://github.com/Oneflow-Inc/libai.git
  cd libai
  ```
2. Create a conda virtual environment and activate it:

  ```bash
  conda create -n libai python=3.7 -y
  conda activate libai
  ```

3. Install the stable release of OneFlow with `CUDA` support:

  ```bash
  python3 -m pip install -f https://release.oneflow.info oneflow==0.6.0+cu102
  ```
  
  For other release, see [OneFlow installation guide](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package).

4. Install other requirements:

  ```bash
  pip install -r requirements.txt
  ```

5. Build the extensional cpp dependency:

  ```bash
  cd /libai/data/data_utils
  make
  ```
