import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import oneflow as flow

flow.mock_torch.enable(lazy=True)
import json
from pathlib import Path
from typing import Dict, List, Optional, TypeVar

import oneflow as torch
import oneflow.nn.functional as F
from lm_eval import evaluator, tasks, utils  # noqa
from lm_eval.api.model import LM  # noqa
from lm_eval.models.utils import chunks  # noqa
from tqdm import tqdm

import libai.utils.distributed as dist  # noqa

T = TypeVar("T")


class EvalHarnessBase(LM):
    def __init__(self, model, tokenizer, model_name, batch_size: int, cfg: dict):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.batch_size_per_gpu = batch_size
        self.cfg = cfg

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        pass

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def max_length(self):
        return self.cfg.max_position_embeddings

    @property
    def vocab_size(self):
        return self.cfg.vocab_size

    @property
    def max_gen_toks(self):
        return self.cfg.get("max_length", 64)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu * dist.get_world_size()

    @property
    def device(self):
        return flow.device("cuda:0")

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def batch_encode(self, strings: List[str]) -> Dict:
        return self.tokenizer.batch_encode_plus(strings, padding=True)

    @flow.inference_mode()
    def _model_call(self, inps):
        inps = inps.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )
        return self.model(inps)["logits"].to_local().to(flow.float32)

    def _model_generate(self, context, max_length, eos_token_id) -> flow.Tensor:
        context = dist.convert_to_distributed_default_setting(context)
        out = self.model.generate(
            context,
            max_length,
            eos_token_id=eos_token_id,
        )
        return out.unsqueeze(0)

    def loglikelihood(self, requests, disable_tqdm=False):
        new_reqs = []
        for request in tqdm(requests, disable=disable_tqdm):
            context, continuation = request.arguments
            if context == "":
                # end of text as context
                context_enc = [self.eos_token_id]
            else:
                context_enc = self.tok_encode(context)

            continuation_enc = self.tok_encode(continuation)[: self.max_length]

            new_reqs.append(((context, continuation), context_enc, continuation_enc))
        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for
            # that
            string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # TODO: automatic (variable) batch size detection for vectorization
        re_ord = utils.Reorderer(requests, _collate)
        for chunk in chunks(tqdm(re_ord.get_reordered(), disable=disable_tqdm), self.batch_size):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_length = None

            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                ).to(self.device)
                (inplen,) = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = padding_length if padding_length is not None else inplen

                # pad length from seq to padding_length
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(
                            inp.device
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                )

                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
            multi_logits = F.log_softmax(
                self._model_call(batched_inps), dim=-1
            ).cpu()  # [batch, padding_length, vocab]

            for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                chunk, multi_logits, inps, inplens, cont_toks_list
            ):

                # Slice to original seq length
                contlen = len(cont_toks)
                logits = logits[inplen - contlen : inplen].unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

        return re_ord.get_original(res)

    def generate_until(self, requests, disable_tqdm=False) -> List[str]:
        res = []

        for chunk in chunks(
            tqdm(requests, disable=disable_tqdm, desc="Running generate_until requests"),
            self.batch_size,
        ):
            _, until = chunk[0].arguments
            if isinstance(until, dict):
                until = until["until"]
            if isinstance(until, str):
                until = [until]
            primary_until = self.tok_encode(until[0])
            reqs = []
            for request in chunk:
                reqs.append(request.arguments[0])
            context_enc = torch.tensor(self.batch_encode(reqs)["input_ids"]).to(self.device)[
                :, self.max_gen_toks - self.max_length :
            ]
            cont = self._model_generate(
                context_enc, context_enc.shape[1] + self.max_gen_toks, primary_until[0]
            )

            for i in range(cont[0].shape[0]):
                s = self.tok_decode(cont[0].tolist()[i][context_enc.shape[1] :])
                for term in until:
                    s = s.split(term)[0]

                res.append(s)
        return res

    @flow.inference_mode()
    def run_eval(
        self,
        eval_tasks: List[str],
        limit: Optional[int],
        bootstrap_iters: int,
    ) -> Dict:
        import fnmatch

        task_manager = tasks.TaskManager()
        all_tasks = task_manager.all_tasks

        def pattern_match(patterns, source_list):
            task_names = set()
            for pattern in patterns:
                for matching in fnmatch.filter(source_list, pattern):
                    task_names.add(matching)
            task_names = list(task_names)
            task_names.sort()
            return task_names

        eval_tasks = pattern_match(eval_tasks, all_tasks)
        print(f"Found tasks: {eval_tasks}")

        if dist.is_main_process() == 0:
            tasks.get_task_dict(eval_tasks)
        dist.synchronize()

        lm = self
        results = evaluator.evaluate(
            lm=lm,
            task_dict=tasks.get_task_dict(task_name_list=eval_tasks),
            limit=limit,
            bootstrap_iters=bootstrap_iters,
        )
        results["config"] = dict(
            model=self.model_name,
            batch_size=self.batch_size,
            device=str(self.device),
            limit=limit,
            bootstrap_iters=bootstrap_iters,
        )
        return results


@flow.inference_mode()
def run_eval_harness(
    model,
    tokenizer,
    model_name,
    eval_tasks: List[str] = [
        "hellaswag",
    ],
    batch_size_per_gpu: int = 1,
    save_filepath: Optional[Path] = None,
    limit: Optional[int] = None,
    bootstrap_iters: int = 100000,
    dtype=flow.float16,
    cfg=None,
):
    model.eval()
    model = model.to(dtype)
    with flow.no_grad():
        eval_harness = EvalHarnessBase(model, tokenizer, model_name, batch_size_per_gpu, cfg)
        results = eval_harness.run_eval(eval_tasks, limit, bootstrap_iters)
    if save_filepath is None:
        print(results["results"])
    else:
        print(f"Saving results to {str(save_filepath)!r}")
        data = json.dumps(results)
        with open(save_filepath, "w") as fw:
            fw.write(data)
