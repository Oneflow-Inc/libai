# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and
# The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Callable, List, Tuple

import oneflow as flow


class LogitsProcessorList(list):
    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor, **kwargs) -> flow.Tensor:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} "
                        "for {processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


class NormalizationLogitsProcessor(object):
    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor) -> flow.Tensor:
        scores = scores.log_softmax(dim=-1)
        return scores


class InfNanRemoveLogitsProcessor(object):
    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor) -> flow.Tensor:
        scores[scores != scores] = 0.0
        scores[scores == float("inf")] = flow.finfo(scores.dtype).max
        return scores


class ForcedEOSTokenLogitsProcessor(object):
    def __init__(self, max_length: int, eos_token_id: int):
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor) -> flow.Tensor:
        cur_len = input_ids.shape[-1]
        if cur_len == self.max_length - 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]] = -float("inf")
            scores[:, self.eos_token_id] = 0
        return scores


class ForcedBOSTokenLogitsProcessor(object):
    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor) -> flow.Tensor:
        cur_len = input_ids.shape[-1]
        if cur_len == 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.bos_token_id]] = -float("inf")
            scores[:, self.bos_token_id] = 0
        return scores


class RepetitionPenaltyLogitsProcessor(object):
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor) -> flow.Tensor:
        score = flow.gather(scores, 1, input_ids)
        score = flow.where(score < 0, score * self.penalty, score / self.penalty)
        scores = flow.scatter(scores, 1, input_ids, score)
        return scores


class HammingDiversityLogitsProcessor(object):
    def __init__(self, diversity_penalty: float, num_beams: int, num_beam_groups: int):
        if not isinstance(diversity_penalty, float) or (not diversity_penalty > 0.0):
            raise ValueError("`diversity_penalty` should be a float strictly larger than 0.")
        self._diversity_penalty = diversity_penalty
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        if num_beam_groups > num_beams:
            raise ValueError("`beam_groups` has to be smaller or equal to `num_beams`.")
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(self, input_ids, scores, current_tokens, beam_group_idx) -> flow.Tensor:
        scores = scores.numpy()

        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]

        if group_start_idx == 0:
            return scores

        for batch_idx in range(batch_size):
            # predicted tokens of last time step of previous groups
            previous_group_tokens = current_tokens[
                batch_idx * self._num_beams : batch_idx * self._num_beams + group_start_idx
            ]
            token_frequency = flow.bincount(previous_group_tokens, minlength=vocab_size)
            scores[batch_idx * group_size : (batch_idx + 1) * group_size] = (
                scores[batch_idx * group_size : (batch_idx + 1) * group_size]
                - self._diversity_penalty * token_frequency
            )

        return scores


def _get_ngrams(ngram_size: int, prev_input_ids: flow.Tensor, num_hypos: int):
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [
                ngram[-1]
            ]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(
    ngram_size: int, prev_input_ids: flow.Tensor, num_hypos: int, cur_len: int
):
    if cur_len + 1 < ngram_size:
        return [[] for _ in range(num_hypos)]

    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)

    banned_tokens = [
        _get_generated_ngrams(
            generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len
        )
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


class NoRepeatNGramLogitsProcessor(object):
    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}"
            )
        self.ngram_size = ngram_size

    def __call__(self, input_ids, scores) -> flow.Tensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = _calc_banned_ngram_tokens(
            self.ngram_size, input_ids, num_batch_hypotheses, cur_len
        )

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores


class EncoderNoRepeatNGramLogitsProcessor(object):
    def __init__(self, encoder_ngram_size: int, encoder_input_ids: flow.Tensor):
        if not isinstance(encoder_ngram_size, int) or encoder_ngram_size <= 0:
            raise ValueError(
                "`encoder_ngram_size` has to be a strictly positive integer, but is "
                f"{encoder_ngram_size}"
            )
        self.ngram_size = encoder_ngram_size
        if len(encoder_input_ids.shape) == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        self.batch_size = encoder_input_ids.shape[0]
        self.generated_ngrams = _get_ngrams(encoder_ngram_size, encoder_input_ids, self.batch_size)

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor) -> flow.Tensor:
        # B x num_beams
        num_hypos = scores.shape[0]
        num_beams = num_hypos // self.batch_size
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = [
            _get_generated_ngrams(
                self.generated_ngrams[hypo_idx // num_beams],
                input_ids[hypo_idx],
                self.ngram_size,
                cur_len,
            )
            for hypo_idx in range(num_hypos)
        ]

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores


class MinLengthLogitsProcessor(object):
    def __init__(self, min_length: int, eos_token_id: int):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor) -> flow.Tensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            scores[:, self.eos_token_id] = -float("inf")
        return scores


class PrefixConstrainedLogitsProcessor(object):
    def __init__(
        self, prefix_allowed_tokens_fn: Callable[[int, flow.Tensor], List[int]], num_beams: int
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor) -> flow.Tensor:
        mask = flow.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(
            input_ids.view(-1, self._num_beams, input_ids.shape[-1])
        ):
            for beam_id, sent in enumerate(beam_sent):
                mask[
                    batch_id * self._num_beams + beam_id,
                    self._prefix_allowed_tokens_fn(batch_id, sent),
                ] = 0

        return scores + mask


class ExponentialDecayLengthPenalty(object):
    def __init__(
        self, exponential_decay_length_penalty: Tuple, eos_token_id: int, input_ids_seq_length: int
    ):
        self.regulation_start = exponential_decay_length_penalty[0] + input_ids_seq_length
        self.regulation_factor = exponential_decay_length_penalty[1]
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor) -> flow.Tensor:
        cur_len = input_ids.shape[-1]
        if cur_len > self.regulation_start:
            scores[:, self.eos_token_id] = scores[:, self.eos_token_id] * pow(
                self.regulation_factor, cur_len - self.regulation_start
            )
        return scores


class TemperatureLogitsWarper(object):
    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(
                f"`temperature` has to be a strictly positive float, but is {temperature}"
            )
        self.temperature = temperature

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor) -> flow.Tensor:
        scores = scores / self.temperature
        return scores


class TopPLogitsWarper(object):
    def __init__(
        self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1
    ):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor) -> flow.Tensor:
        sorted_logits, sorted_indices = flow.sort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1
            # because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = flow.scatter(
            sorted_indices_to_remove, 1, sorted_indices, sorted_indices_to_remove
        )
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopKLogitsWarper(object):
    def __init__(
        self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1
    ):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor) -> flow.Tensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < flow.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TypicalLogitsWarper(object):
    def __init__(
        self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1
    ):
        mass = float(mass)
        if not (mass > 0 and mass < 1):
            raise ValueError(f"`typical_p` has to be a float > 0 and < 1, but is {mass}")

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor) -> flow.Tensor:

        # calculate entropy
        normalized = flow.nn.functional.log_softmax(scores, dim=-1)
        p = flow.exp(normalized)
        ent = -flow.nansum(normalized * p, dim=-1, keepdim=True)

        # shift and sort
        shifted_scores = flow.abs((-normalized) - ent)
        sorted_scores, sorted_indices = flow.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind[last_ind < 0] = 0
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            # (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = flow.scatter(
            sorted_indices_to_remove, 1, sorted_indices, sorted_indices_to_remove
        )
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
