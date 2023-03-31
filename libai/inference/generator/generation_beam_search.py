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

import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from typing import Optional, Tuple

import oneflow as flow

from libai.utils import distributed as dist


class BeamScorer(ABC):
    @abstractmethod
    def process(
        self,
        input_ids: flow.Tensor,
        next_scores: flow.Tensor,
        next_tokens: flow.Tensor,
        next_indices: flow.Tensor,
        **kwargs,
    ):
        raise NotImplementedError("This is an abstract method.")


class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self) -> int:
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(
        self, hyp: flow.Tensor, sum_logprobs: float, beam_indices: Optional[flow.Tensor] = None
    ):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class BeamSearchScorer(BeamScorer):
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        **kwargs,
    ):
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]

        self._done = flow.tensor(
            [False for _ in range(batch_size)],
            dtype=flow.bool,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=flow.placement("cuda", list(range(dist.get_world_size()))),
        )

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}."
                "For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

        if (
            not isinstance(num_beam_groups, int)
            or (num_beam_groups > num_beams)
            or (num_beams % num_beam_groups != 0)
        ):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than `num_beams` and "
                f"`num_beams` has to be divisible by `num_beam_groups`, but is {num_beam_groups}"
                f"with `num_beams` being {num_beams}."
            )

        if "max_length" in kwargs:
            warnings.warn(
                "Passing `max_length` to BeamSearchScorer is deprecated and has no effect. "
                "`max_length` should be passed directly to `beam_search(...)`, `beam_sample(...)`"
                ", or `group_beam_search(...)`."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: flow.Tensor,
        next_scores: flow.Tensor,
        next_tokens: flow.Tensor,
        next_indices: flow.Tensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        beam_indices: Optional[flow.Tensor] = None,
    ) -> Tuple[flow.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group "
                    f"beam size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )
        next_beam_scores = flow.zeros(
            (batch_size, self.group_size),
            dtype=next_scores.dtype,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=flow.placement("cuda", list(range(dist.get_world_size()))),
        )
        next_beam_tokens = flow.zeros(
            (batch_size, self.group_size),
            dtype=next_tokens.dtype,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=flow.placement("cuda", list(range(dist.get_world_size()))),
        )
        next_beam_indices = flow.zeros(
            (batch_size, self.group_size),
            dtype=next_indices.dtype,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=flow.placement("cuda", list(range(dist.get_world_size()))),
        )

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(
                        f"Batch can only be done if at least {self.num_beams} beams have "
                        "been generated"
                    )
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError(
                        "Generated beams >= num_beams -> eos_token_id and pad_token have "
                        "to be defined"
                    )
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    if beam_indices is not None:
                        beam_index = beam_indices[batch_beam_idx]
                        beam_index = beam_index + (next_index,)
                    else:
                        beam_index = None

                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        beam_indices=beam_index,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal "
                    f"to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} "
                    "are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        input_ids: flow.Tensor,
        final_beam_scores: flow.Tensor,
        final_beam_tokens: flow.Tensor,
        final_beam_indices: flow.Tensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        beam_indices: Optional[flow.Tensor] = None,
    ):
        batch_size = len(self._beam_hyps)
        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
                beam_hyp.add(final_tokens, final_score, beam_indices=beam_index)

        # select the best hypotheses
        sent_lengths = flow.zeros(
            batch_size * self.num_beam_hyps_to_keep,
            dtype=flow.long,
            sbp=input_ids.sbp,
            placement=input_ids.placement,
        )
        best = []
        best_indices = []
        best_scores = flow.zeros(
            batch_size * self.num_beam_hyps_to_keep,
            dtype=flow.float32,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=flow.placement("cuda", list(range(dist.get_world_size()))),
        )

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_index = best_hyp_tuple[2]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append hyp to lists
                best.append(best_hyp)

                # append indices to list
                best_indices.append(best_index)

                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = (
            min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        )
        decoded = flow.zeros(
            (batch_size * self.num_beam_hyps_to_keep, sent_max_len),
            dtype=flow.long,
            sbp=input_ids.sbp,
            placement=input_ids.placement,
        )

        if len(best_indices) > 0 and best_indices[0] is not None:
            indices = flow.zeros(
                (batch_size * self.num_beam_hyps_to_keep, sent_max_len),
                dtype=flow.long,
                sbp=input_ids.sbp,
                placement=input_ids.placement,
            )
        else:
            indices = None

        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        if indices is not None:
            indices.fill_(-1)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            decoded[i, : sent_lengths[i]] = hypo

            if indices is not None:
                indices[i, : len(best_idx)] = flow.tensor(best_idx)

            if sent_lengths[i] < sent_max_len:
                decoded[i, sent_lengths[i]] = eos_token_id

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }
        )
