from libai.config import LazyConfig, LazyCall
from libai.models.utils.model_loader.base_loader import ModelLoaderLiBai,ModelLoaderHuggerFace  # noqa
from projects.Llama.llama import LlamaForCausalLM  # noqa
import libai.utils.distributed as dist  # noqa
import json
from transformers import AutoTokenizer as HF_AutoTokenizer
import importlib

class LLMLoaderLibai(ModelLoaderLiBai):
    def __init__(self, model, libai_cfg, pretrained_model_path,base_model_prefix,**kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_2 = base_model_prefix

def get_special_arguments(cfg):
    with open('./projects/Eval_LLM/special_arguments.json','r') as f:
        arguments = json.load(f)
    special_arguments=arguments[cfg.eval_config.model_type]
    return special_arguments

def main():
    cfg = LazyConfig.load('./projects/Eval_LLM/config.py')
    dist.setup_dist_util(cfg.parallel_config)
    special_arguments = get_special_arguments(cfg)
    print('Loading Model...')
    model_cfg = LazyConfig.load(special_arguments['config_path'])
    if model_cfg.cfg.max_position_embeddings is None:
        model_cfg.cfg.max_position_embeddings = 1024

    model_class = getattr(importlib.import_module(special_arguments['model_class_prefix']),special_arguments['model_class'])

    assert cfg.eval_config.model_weight_type in ['huggingface','libai'], 'model_weight_type must be huggingface or libai'
    if cfg.eval_config.model_weight_type=='huggingface':
        huggingface_loader = getattr(importlib.import_module(special_arguments['huggingface_loader_prefix']),special_arguments['huggingface_loader'])
        load_func = huggingface_loader(
            model=model_class,
            libai_cfg=model_cfg.cfg,
            pretrained_model_path=cfg.eval_config.pretrained_model_path,
        )
    else:
        load_func = LLMLoaderLibai(
            model=model_class,
            libai_cfg=model_cfg.cfg,
            pretrained_model_path=cfg.eval_config.pretrained_model_path,
            base_model_prefix=special_arguments['base_model_prefix_2']
        )
    
    tokenizer=HF_AutoTokenizer.from_pretrained(cfg.eval_config.hf_tokenizer_path,trust_remote_code=True)
    with open(cfg.eval_config.hf_tokenizer_path+'/config.json','r') as f:
        generation_config = json.load(f)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = generation_config['pad_token_id']
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = generation_config['eos_token_id']
    model = load_func.load()
    print('Model Loaded!')

    from projects.Eval_LLM.eval_harness import run_eval_harness  # noqa
    run_eval_harness(model, tokenizer, cfg.eval_config.model_type, eval_tasks=cfg.eval_config.eval_tasks, batch_size_per_gpu=cfg.eval_config.batch_size_per_gpu, cfg=model_cfg.cfg)

if __name__ == "__main__":
    main()
