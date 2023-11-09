from libai.config import instantiate
from projects.Llama.configs.llama_config import cfg, tokenizer
from projects.Llama.llama import LlamaForCausalLM
from projects.Llama.llama_loader import LlamaLoaderHuggerFace
import libai.utils.distributed as dist
from omegaconf import DictConfig


pretrained_model_path = "/data/hf_models/NousResearch/Llama-2-7b-chat-hf"
text = "a dog is flying on the sky"


if __name__ == "__main__":
    parallel_config = DictConfig(
            dict(
                data_parallel_size=1,
                tensor_parallel_size=2,
                pipeline_parallel_size=2,  # set to 1, unsupport pipeline parallel now
                pipeline_num_layers=32,
                device_type="cuda",
            )
        )
    dist.setup_dist_util(parallel_config)

    tokenizer = instantiate(tokenizer)
    input_ids = tokenizer.tokenize(text, add_bos=True)

    load_func = LlamaLoaderHuggerFace(
        model=LlamaForCausalLM,
        libai_cfg=cfg,
        pretrained_model_path=pretrained_model_path,
    )
    model = load_func.load()
    model.eval()

    tokens = model.generate(
        input_ids,
        max_new_tokens=20,
        do_sample=False,
    )

    if dist.is_main_process():
        res = tokenizer.decode(tokens)
        print(res)
