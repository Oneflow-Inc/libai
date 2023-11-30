from omegaconf import DictConfig

import libai.utils.distributed as dist
from libai.config import instantiate
from projects.Llama.configs.llama_config import cfg, tokenization
from projects.Llama.llama import LlamaForCausalLM
from projects.Llama.utils.llama_loader import LlamaLoaderHuggerFace, LlamaLoaderLiBai

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

    tokenizer = instantiate(tokenization.tokenizer)
    input_ids = tokenizer.tokenize(text, add_bos=True, padding=True)
    load_func = LlamaLoaderHuggerFace(
        model=LlamaForCausalLM,
        libai_cfg=cfg,
        pretrained_model_path="/data/home/xiezipeng/meta-llama/Llama-2-7b-hf/",
    )
    model_hf = load_func.load()
    model_hf.eval()
    res_hf = model_hf(input_ids)
    print(res_hf)

    load_func = LlamaLoaderLiBai(
        model=LlamaForCausalLM,
        libai_cfg=cfg,
        pretrained_model_path="/data/home/xiezipeng/libai/sft_result/model_0000399/model",
    )
    model_libai = load_func.load()
    model_libai.eval()
    res_libai = model_libai(input_ids)
    print(res_libai)
