import json
from pathlib import Path
from typing import Dict, List, Optional

import oneflow as flow
flow.mock_torch.enable(lazy=True)

from lm_eval import base, evaluator, tasks
from lm_eval.base import BaseLM

from omegaconf import DictConfig

import libai.utils.distributed as dist
from libai.config import instantiate
from projects.Llama.configs.llama_config import cfg, tokenization
from projects.Llama.llama import LlamaForCausalLM
from projects.Llama.utils.llama_loader import LlamaLoaderHuggerFace, LlamaLoaderLiBai


class EvalHarnessBase(BaseLM):
    def __init__(self, model, tokenizer, batch_size: int, cfg: dict):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size_per_gpu = batch_size
        self.cfg = cfg

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        pass

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.cfg.max_position_embeddings

    @property
    def vocab_size(self):
        return self.cfg.vocab_size

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu * dist.get_world_size()
    
    @property
    def device(self):
        return flow.device('cuda:0')

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.tokenize(string, add_bos=False, add_eos=False).squeeze(0).tolist()

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    @flow.inference_mode()
    def _model_call(self, inps):
        inps = inps.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )
        return self.model(inps)["logits"].to_local()

    def _model_generate(self, context, max_length, eos_token_id) -> flow.Tensor:
        # this only supports batch size 1
        assert context.shape[0] == 1
        out = self.model.generate(context[0], max_length, eos_id=eos_token_id)
        return out.unsqueeze(0)

    @flow.inference_mode()
    def run_eval(
        self, eval_tasks: List[str], num_fewshot: int, limit: Optional[int], bootstrap_iters: int,
    ) -> Dict:
        import fnmatch

        def pattern_match(patterns, source_list):
            task_names = set()
            for pattern in patterns:
                for matching in fnmatch.filter(source_list, pattern):
                    task_names.add(matching)
            return list(task_names)

        eval_tasks = pattern_match(eval_tasks, tasks.ALL_TASKS)
        print(f"Found tasks: {eval_tasks}")

        if dist.is_main_process() == 0:
            tasks.get_task_dict(eval_tasks)
        dist.synchronize()
        tasks.get_task_dict(eval_tasks)

        lm = self

        results = evaluator.evaluate(
            lm=lm,
            task_dict=tasks.get_task_dict(eval_tasks),
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
        )
        results["config"] = dict(
            model="llama",
            batch_size=self.batch_size,
            device=str(self.device),
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
        )
        return results


@flow.inference_mode()
def run_eval_harness(
    model,
    tokenizer,
    eval_tasks: List[str] = ["arc_challenge", "piqa", "hellaswag", "hendrycksTest-*"],
    save_filepath: Optional[Path] = None,
    num_fewshot: int = 0,
    limit: Optional[int] = None,
    bootstrap_iters: int = 100000,
    cfg = None,
):
    model.eval()
    eval_harness = EvalHarnessBase(model, tokenizer, 1, cfg)

    results = eval_harness.run_eval(eval_tasks, num_fewshot, limit, bootstrap_iters)
    if save_filepath is None:
        print(results)
    else:
        print(f"Saving results to {str(save_filepath)!r}")
        data = json.dumps(results)
        with open(save_filepath, "w") as fw:
            fw.write(data)


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
    load_func = LlamaLoaderHuggerFace(
        model=LlamaForCausalLM,
        libai_cfg=cfg,
        pretrained_model_path="/data/home/xiezipeng/hf_models/meta-llama/Llama-2-7b-hf/",
    )
    model = load_func.load()
    run_eval_harness(model, tokenizer, cfg=cfg)
