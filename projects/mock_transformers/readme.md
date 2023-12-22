# Mock transformers

This is an application of mock [transformers](https://github.com/huggingface/transformers), which can perform distributed inference in LiBai with model under the transformers.

**Supported Model**

- [BLOOM](#distributed-infer-bloom): tensor parallel
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

## Distributed Inference Through Mock Transformers

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

## Examples

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
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m") # change your model type 125m~66b
    model._apply(dist.convert_to_distributed_default_setting)
    # initial tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False) # change your model type  125m~66b

```