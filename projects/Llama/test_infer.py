from omegaconf import DictConfig

import libai.utils.distributed as dist
from libai.config import instantiate
from projects.Llama.configs.llama_config import cfg, tokenizer
from projects.Llama.llama import LlamaForCausalLM
from projects.Llama.utils.llama_loader import LlamaLoaderHuggerFace

pretrained_model_path = "/data/hf_models/Llama-2-7b-hf"
text = [
    "a dog is flying on the sky",
    "Wikipedia is a free online",
    "what is beam search?",
]


if __name__ == "__main__":
    parallel_config = DictConfig(
        dict(
            data_parallel_size=1,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            pipeline_num_layers=32,
            device_type="cuda",
        )
    )
    dist.setup_dist_util(parallel_config)

    tokenizer = instantiate(tokenizer)
    input_ids = tokenizer.tokenize(text, add_bos=True, padding=True)
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

    # if dist.is_main_process():
    res = tokenizer.decode(tokens)
    print(res)
