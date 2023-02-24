# flake8: noqa
# black: skip-string-normalization
import init_env
import oneflow as flow
from omegaconf import DictConfig
from oneflow.utils.global_view import global_mode
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer

from libai.layers import Linear
from libai.utils import distributed as dist

# ------replace attention to libai------
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
OPTAttention = LiBaiOPTAttention

# ----------replace Decoder to libai -----
temp_class = OPTDecoderLayer
class LiBaiOPTDecoderLayer(temp_class):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = args[0]
        self.fc1 = Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias, parallel="col")
        self.fc2 = Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias, parallel="row")
OPTDecoderLayer = LiBaiOPTDecoderLayer

if __name__ == "__main__":
    # set dist config
    parallel_config = DictConfig(
        dict(
            data_parallel_size=1,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,  # set to 1, unsupport pipeline parallel now
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
        placement=dist.get_layer_placement(0),
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
