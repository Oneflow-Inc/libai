# Installation

LiBai provides an editable installation way for you to develop your own project based on LiBai's framework.

## Build LiBai from Source

- Clone this repo:

```bash
git clone https://github.com/Oneflow-Inc/libai.git
cd libai
```

- Create a conda virtual environment and activate it:

```bash
conda create -n libai python=3.8 -y
conda activate libai
```

- Install the stable release of OneFlow with `CUDA` support. See [OneFlow installation guide](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package). To use **latest** LiBai(branch `main`), we highly recommend you install **Nightly** Oneflow

  - Stable

    ```bash
    python3 -m pip install --find-links https://release.oneflow.info oneflow==0.9.0+[PLATFORM]
    ```

  - Nightly

    ```
    python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/[PLATFORM]
    ```

  - All available `[PLATFORM]`:
  
    <table class="docutils">
    <thead>
    <tr class="header">
    <th>Platform</th>
    <th>CUDA Driver Version</th>
    <th>Supported GPUs</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td>cu117</td>
    <td>&gt;= 450.80.02</td>
    <td>GTX 10xx, RTX 20xx, A100, RTX 30xx</td>
    </tr>
    <tr class="even">
    <td>cu102</td>
    <td>&gt;= 440.33</td>
    <td>GTX 10xx, RTX 20xx</td>
    </tr>
    <tr class="odd">
    <td>cpu</td>
    <td>N/A</td>
    <td>N/A</td>
    </tr>
    </tbody>
    </table></li>

- Install `pybind11`:

```bash
pip install pybind11
```

- For an editable installation of LiBai:

```bash
pip install -e .
```
