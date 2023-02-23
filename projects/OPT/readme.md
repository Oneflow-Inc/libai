# OPT

This is an reimplement of [OPT](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) inference in LiBai

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


## Environment 

Before running the scripts, make sure to install the library's training dependencies:

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

## distributed infer

for `tensor_parallel=2`, run command in `libai_root`
```
bash tools/infer.sh projects/OPT/dist_infer_opt.py 2
```
The infer code is very simple:
```python
from omegaconf import DictConfig
import init_env

import oneflow as flow
from oneflow.utils.global_view import global_mode
from libai.utils import distributed as dist
from libai.layers import Linear

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer

# -------------------change Linear layer to libai.layers.Linear-------
temp_class = OPTAttention
class LiBaiOPTAttention(temp_class):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim = kwargs["embed_dim"]
        bias = kwargs["bias"]
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias, parallel="col")
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias, parallel="col")
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, parallel="col")
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, parallel="row")
OPTAttention=LiBaiOPTAttention

temp_class = OPTDecoderLayer
class LiBaiOPTDecoderLayer(temp_class):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config=args[0]
        self.fc1 = Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias, parallel="col")
        self.fc2 = Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias, parallel="row")
OPTDecoderLayer=LiBaiOPTDecoderLayer
# ----------------- change end ------------------------------

if __name__ == "__main__":
    # set dist config
    parallel_config = DictConfig(
        dict(
            data_parallel_size=1,
            tensor_parallel_size=2,
            pipeline_parallel_size=1, # set to 1, unsupport pipeline parallel now
            pipeline_num_layers=None,
            )
    )
    dist.setup_dist_util(parallel_config)

    # initial and load model
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").half()
    model._apply(dist.convert_to_distributed_default_setting)
    # initial tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)

    # get input_ids
    prompt = "Hello, I'm am conscious and"
    input_ids = tokenizer(prompt, return_tensors="np").input_ids
    input_ids = flow.from_numpy(input_ids)
    input_ids = input_ids.to_global(
        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        placement=dist.get_layer_placement(0)
    )

    # generate id
    placement_sbp_dict = dict(
        placement=flow.env.all_device_placement("cuda"),
        sbp=flow.sbp.broadcast,
    )
    with global_mode(True, **placement_sbp_dict):
        generated_ids = model.generate(input_ids, max_length=30)
    out_put_ids = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(out_put_ids)
```