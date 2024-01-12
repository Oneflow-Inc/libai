import json
from pathlib import Path
from typing import Dict, List, Optional

import oneflow as flow

flow.mock_torch.enable()

from lm_eval import evaluator, tasks  # noqa v 0.3.0
from lm_eval.base import BaseLM  # noqa
from omegaconf import DictConfig  # noqa

import libai.utils.distributed as dist  # noqa
from libai.config import instantiate  # noqa
from projects.ChatGLM.chatglm import ChatGLMForConditionalGeneration  # noqa
from projects.ChatGLM.configs.chatglm_config import cfg, tokenization  # noqa
from projects.ChatGLM.utils.chatglm_loader import (  # noqa
    ChatGLMLoaderHuggerFace,
    ChatGLMLoaderLiBai,
)


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
        return self.cfg.seq_length

    @property
    def vocab_size(self):
        return self.cfg.vocab_size

    @property
    def max_gen_toks(self):
        return self.cfg.get("max_length", 256)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu * dist.get_world_size()

    @property
    def device(self):
        return flow.device("cuda:0")

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.get_prefix_tokens() + self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(string)
        )

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    @flow.inference_mode()
    def _model_call(self, inps):
        inps = inps.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )
        return self.model(inps)["logits"].to_local().to(flow.float32)

    def _model_generate(self, context, max_length, eos_token_id) -> flow.Tensor:
        # this only supports batch size 1
        assert context.shape[0] == 1
        out = self.model.generate(context[0], max_length, eos_id=eos_token_id)
        return out.unsqueeze(0)

    @flow.inference_mode()
    def run_eval(
        self,
        eval_tasks: List[str],
        num_fewshot: int,
        limit: Optional[int],
        bootstrap_iters: int,
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
            model="chatglm",
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
    eval_tasks: List[str] = [
        "hellaswag",
    ],
    save_filepath: Optional[Path] = None,
    num_fewshot: int = 0,
    limit: Optional[int] = None,
    bootstrap_iters: int = 100000,
    dtype=flow.float16,
    cfg=None,
):
    model.eval()
    model = model.to(dtype)
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
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            pipeline_num_layers=28,
            device_type="cuda",
        )
    )
    dist.setup_dist_util(parallel_config)

    tokenizer = instantiate(tokenization.tokenizer)

    # ----- load huggingface checkpoint -----
    # load_func = ChatGLMLoaderHuggerFace(
    #     model=ChatGLMForConditionalGeneration,
    #     libai_cfg=cfg,
    #     pretrained_model_path="",
    # )

    # ----- load oneflow checkpoint -----
    load_func = ChatGLMLoaderLiBai(
        model=ChatGLMForConditionalGeneration,
        libai_cfg=cfg,
        pretrained_model_path="LiBaiChatGLMModelPath",
    )
    model = load_func.load()
    run_eval_harness(model, tokenizer, cfg=cfg)
