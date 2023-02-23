import oneflow as flow
from libai.utils import distributed as dist
from omegaconf import DictConfig
from oneflow.utils.global_view import global_mode
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

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