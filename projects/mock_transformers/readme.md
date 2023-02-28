# Mock transformers

This is an application of mock [transformers](https://github.com/huggingface/transformers), which can perform distributed inference in LiBai with model under the transformers.

**Supported Model**

- [OPT](#distributed-infer-opt): tensor parallel


## Environment 

Before running the scripts, make sure to install the library's dependencies:

### Install libai

libai installation, refer to [Installation instructions](https://libai.readthedocs.io/en/latest/tutorials/get_started/Installation.html)

```bash
# create conda env
conda create -n libai python=3.8 -y
conda activate libai

# install oneflow nightly, [PLATFORM] could be cu117 or cu102
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/[PLATFORM]

# install libai
git clone https://github.com/Oneflow-Inc/libai.git
cd libai
pip install pybind11
pip install -e .
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

### Install transformers

refer to [transformers installation](https://github.com/huggingface/transformers#installation)
```
python3 -m pip install "transformers>=4.26"
```

Notes

- You need to register a Hugging Face account token and login with `huggingface-cli login`

```bash
python3 -m pip install huggingface_hub
```

- If no command available in the PATH, it might be in the `$HOME/.local/bin`

```bash
 ~/.local/bin/huggingface-cli login
```

## distributed infer OPT

An reimplement of [OPT](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) distributed inference in LiBai

<table class="docutils">
  <tbody>
    <tr>
      <th width="80"> opt inference </th>
      <th valign="bottom" align="left" width="120">Tensor Parallel</th>
      <th valign="bottom" align="left" width="120">Pipeline Parallel</th>
    </tr>
    <tr>
      <td align="left"> <b> Support </b> </td>
      <td align="left">&#10004;</td>
      <td align="left">-</td>
    </tr>
  </tbody>
</table>

for `tensor_parallel=2`, run command in `libai_root`
```
bash tools/infer.sh projects/mock_transformers/dist_infer_opt.py 2
```
modify the infer code `dist_infer_opt.py` according to your own needs:
```python
...

if __name__ == "__main__":
    # set dist config
    parallel_config = DictConfig(
        dict(
            data_parallel_size=1,
            tensor_parallel_size=2, # modify it according to your own needs
            pipeline_parallel_size=1, # set to 1, unsupport pipeline parallel now
            pipeline_num_layers=None,
            )
    )
    dist.setup_dist_util(parallel_config)

    ...
    # initial and load model
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").half() # change your model type 125m~66b
    model._apply(dist.convert_to_distributed_default_setting)
    # initial tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False) # change your model type  125m~66b

```