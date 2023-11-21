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
import logging
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import oneflow as flow
from oneflow import nn

from libai.utils import distributed as dist

from .generation_beam_search import BeamScorer, BeamSearchScorer
from .generation_logits_processor import (
    EncoderNoRepeatNGramLogitsProcessor,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    NormalizationLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from .generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

logger = logging.getLogger(__name__)


class Generator:
    def _prepare_model_inputs(
        self,
        inputs: Optional[flow.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, flow.Tensor]] = None,
    ):
        if self.cfg.is_encoder_decoder:
            input_name = "encoder_input_ids"
        else:
            input_name = "input_ids"

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside "
                f"{input_name} which is not allowed."
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        if inputs is None:
            inputs = self._prepare_input_ids_for_generation(
                bos_token_id, model_kwargs.get("encoder_outputs", None)
            )
        return inputs, input_name, model_kwargs

    def prepare_inputs_for_generation(self, input_ids: flow.Tensor, **kwargs):
        """
        Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the
        generate method.
        """
        return {"input_ids": input_ids}

    def _prepare_input_ids_for_generation(
        self, bos_token_id: Optional[int], encoder_outputs: Optional[flow.Tensor]
    ):
        if self.cfg.is_encoder_decoder and encoder_outputs is not None:
            shape = encoder_outputs.size()[:-1]
            return (
                flow.ones(
                    shape,
                    dtype=flow.long,
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                    placement=dist.get_layer_placement(0),
                )
                * -100
            )

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")
        return (
            flow.ones(
                (1, 1),
                dtype=flow.long,
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(0),
            )
            * bos_token_id
        )

    def _prepare_attention_mask_for_generation(
        self,
        inputs: flow.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[int],
    ):
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [
            flow.int64,
            flow.long,
        ]
        is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        # Check if input is input_ids and padded -> only then is attention_mask defined
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return inputs.ne(pad_token_id).bool()
        else:
            return flow.ones(
                inputs.shape[:2],
                dtype=flow.bool,
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(0),
            )

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: flow.Tensor, model_kwargs, model_input_name: str
    ):
        only_encoder = True
        model_kwargs[model_input_name] = inputs_tensor
        if "encoder_decoder_attn_mask" in set(inspect.signature(self.forward).parameters):
            model_kwargs["encoder_decoder_attn_mask"] = model_kwargs["encoder_attn_mask"]
        model_kwargs["encoder_outputs"] = self(**model_kwargs, only_encoder=only_encoder)
        model_kwargs.pop(model_input_name)
        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        model_kwargs=None,
    ):
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            return model_kwargs.pop("decoder_input_ids")
        else:
            decoder_start_token_id = (
                decoder_start_token_id
                if decoder_start_token_id
                else self.cfg.decoder_start_token_id
            )
            return (
                flow.ones(
                    (batch_size, 1),
                    dtype=flow.long,
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                    placement=dist.get_layer_placement(0),
                )
                * decoder_start_token_id
            )

    def _get_decoder_start_token_id(
        self, decoder_start_token_id: int = None, bos_token_id: int = None
    ):
        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif self.cfg.is_encoder_decoder:
            return self.cfg.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        else:
            return self.cfg.bos_token_idx

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: flow.Tensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[flow.Tensor] = None,
        encoder_outputs: Optional[flow.Tensor] = None,
        **model_kwargs,
    ):
        expanded_return_idx = (
            flow.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1)
        )
        expanded_return_idx = expanded_return_idx.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )

        input_ids = input_ids.index_select(0, expanded_return_idx)

        # token_type ids not supported.

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            encoder_outputs = encoder_outputs.to_global(placement=expanded_return_idx.placement)
            encoder_outputs = encoder_outputs.index_select(0, expanded_return_idx)
            model_kwargs["encoder_outputs"] = encoder_outputs
            model_kwargs["encoder_attn_mask"] = model_kwargs["encoder_attn_mask"].index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_decoder_attn_mask"] = model_kwargs["encoder_attn_mask"]
        return input_ids, model_kwargs

    def _update_model_kwargs_for_generation(
        self, outputs, model_kwargs, is_encoder_decoder: bool = False
    ):
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs["past_key_values"]
        elif "mems" in outputs:
            model_kwargs["past"] = outputs["mems"]
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs["past_buckets_states"]
        elif self.past_key_values[-1] is not None:
            model_kwargs["past"] = self.past_key_values
        else:
            model_kwargs["past"] = None

        # update attention mask
        if "attention_mask" in model_kwargs and not is_encoder_decoder:
            attention_mask = model_kwargs["attention_mask"]
            pad = flow.ones(
                (attention_mask.shape[0], 1),
                sbp=attention_mask.sbp,
                placement=attention_mask.placement,
            )
            model_kwargs["attention_mask"] = flow.cat([attention_mask, pad], dim=-1)

        if "decoder_attn_mask" in model_kwargs and is_encoder_decoder:
            attention_mask = model_kwargs["decoder_attn_mask"]
            pad = flow.ones(
                (attention_mask.shape[0], 1),
                sbp=attention_mask.sbp,
                placement=attention_mask.placement,
            )
            model_kwargs["decoder_attn_mask"] = flow.cat([attention_mask, pad], dim=-1)

        return model_kwargs

    def _reorder_cache(self, past, beam_idx):
        raise NotImplementedError(
            "Make sure that a `_reorder_cache` function is correctly implemented in "
            f"{self.__class__.__module__} to enable beam search for {self.__class__}"
        )

    def _get_logits_warper(
        self,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        temperature: Optional[float] = None,
        num_beams: Optional[int] = None,
        renormalize_logits: Optional[bool] = None,
    ):
        # instantiate warpers list
        warpers = LogitsProcessorList()

        # all samplers can be found in `generation_utils_samplers.py`
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(
                TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1))
            )
        if top_p is not None and top_p < 1.0:
            warpers.append(
                TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1))
            )
        if typical_p is not None and typical_p < 1.0:
            warpers.append(
                TypicalLogitsWarper(mass=typical_p, min_tokens_to_keep=(2 if num_beams > 1 else 1))
            )
        # `LogitNormalization` should always be the last logit processor, when present
        if renormalize_logits is True:
            warpers.append(NormalizationLogitsProcessor())
        return warpers

    def _get_logits_processor(
        self,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        encoder_no_repeat_ngram_size: int,
        input_ids_seq_length: int,
        encoder_input_ids: flow.Tensor,
        min_length: int,
        max_length: int,
        eos_token_id: int,
        forced_bos_token_id: int,
        forced_eos_token_id: int,
        prefix_allowed_tokens_fn: Callable[[int, flow.Tensor], List[int]],
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        remove_invalid_values: bool,
        exponential_decay_length_penalty: Tuple,
        logits_processor: Optional[LogitsProcessorList],
        renormalize_logits: Optional[bool],
    ):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant
        [`LogitsProcessor`] instances used to modify the scores of the language model head.
        """
        processors = LogitsProcessorList()

        # instantiate processors list
        if diversity_penalty is not None and diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=diversity_penalty,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                )
            )
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
            if self.cfg.is_encoder_decoder:
                processors.append(
                    EncoderNoRepeatNGramLogitsProcessor(
                        encoder_no_repeat_ngram_size, encoder_input_ids
                    )
                )
            else:
                raise ValueError(
                    "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only "
                    "architecture"
                )
        if min_length is not None and eos_token_id is not None and min_length > 0:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if prefix_allowed_tokens_fn is not None:
            processors.append(
                PrefixConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn, num_beams // num_beam_groups
                )
            )
        if forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
        if remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())
        if exponential_decay_length_penalty is not None:
            processors.append(
                ExponentialDecayLengthPenalty(
                    exponential_decay_length_penalty, eos_token_id, input_ids_seq_length
                )
            )
        processors = self._merge_criteria_processor_list(processors, logits_processor)
        # `LogitNormalization` should always be the last logit processor, when present
        if renormalize_logits is True:
            processors.append(NormalizationLogitsProcessor())
        return processors

    def _get_stopping_criteria(
        self,
        max_length: Optional[int],
        max_time: Optional[float],
        stopping_criteria: Optional[StoppingCriteriaList],
    ):
        criteria = StoppingCriteriaList()
        if max_length is not None:
            criteria.append(MaxLengthCriteria(max_length=max_length))
        if max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=max_time))
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria

    def _merge_criteria_processor_list(self, default_list, custom_list):
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    raise ValueError("Criteria repetition error.")
        default_list.extend(custom_list)
        return default_list

    def compute_transition_beam_scores(
        self,
        sequences: flow.Tensor,
        scores: Tuple[flow.Tensor],
        beam_indices: flow.Tensor,
        eos_token_id: int = None,
    ):
        scores = flow.stack(scores).reshape(len(scores), -1).transpose(0, 1)

        beam_indices_mask = beam_indices < 0
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
        beam_indices = beam_indices[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]

        beam_indices[beam_indices_mask] = 0

        beam_sequence_indices = beam_indices * self.cfg.vocab_size

        cut_idx = sequences.shape[-1] - max_beam_length
        indices = sequences[:, cut_idx:] + beam_sequence_indices

        transition_scores = scores.gather(0, indices)

        transition_scores[beam_indices_mask] = 0

        return transition_scores

    def _validate_model_kwargs(self, model_kwargs):
        if self.cfg.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)
        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        if "kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)
        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} "
                "(note: typos in the generate arguments will also show up in this list)"
            )

    def greedy_search(
        self,
        input_ids: flow.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        is_encoder_decoder: bool = False,
        output_scores: bool = False,
        **model_kwargs,
    ):
        pad_token_id = pad_token_id if pad_token_id is not None else self.cfg.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id
        output_scores = output_scores if output_scores is not None else self.cfg.output_scores
        scores = () if output_scores else None
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use MaxLengthCriteria" " instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        # keep track of which sequences are already finished
        unfinished_sequences = flow.ones(input_ids.shape[0])
        cur_len = input_ids.shape[-1]
        while True:
            if input_ids.size(0) > 1:
                input_ids = input_ids.to_global(
                    sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast])
                )

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # generate
            outputs = self(**model_inputs)
            next_token_logits = outputs["logits"][:, -1, :]

            # logits_processor
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores
            if output_scores:
                scores += (next_token_scores,)

            # argmax
            next_tokens = flow.argmax(next_token_scores, dim=-1)
            next_tokens = next_tokens.to_global(placement=input_ids.placement)
            unfinished_sequences = unfinished_sequences.to_global(
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=input_ids.placement,
            )

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            next_tokens = next_tokens.to(flow.long)
            input_ids = flow.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = flow.mul(
                    unfinished_sequences, (next_tokens != eos_token_id).long()
                )

            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

        # Release records
        if "past_key_values" in self.__dir__():
            self.past_key_values = [None] * self.cfg.hidden_layers
        if "encoder_states" in self.__dir__():
            self.encoder_states = None

        return input_ids

    def multinomial_sample(
        self,
        input_ids: flow.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        is_encoder_decoder: bool = False,
        output_scores: bool = False,
        **model_kwargs,
    ):
        # init values
        pad_token_id = pad_token_id if pad_token_id is not None else self.cfg.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id
        output_scores = output_scores if output_scores is not None else self.cfg.output_scores
        scores = () if output_scores else None
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use "
                "`stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))`"
                "instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()

        unfinished_sequences = flow.ones(input_ids.shape[0])
        cur_len = input_ids.shape[-1]

        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # generate
            outputs = self(**model_inputs)
            next_token_logits = outputs["logits"][:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores
            if output_scores:
                scores += (next_token_scores,)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            probs = probs.to_global(
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(0),
            ).to_local()
            next_tokens = flow.multinomial(probs, num_samples=1).squeeze(1)
            next_tokens = next_tokens.to_global(
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(0),
            )
            unfinished_sequences = unfinished_sequences.to_global(
                sbp=next_tokens.sbp, placement=next_tokens.placement
            )

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            next_tokens = next_tokens.to(flow.long)
            input_ids = flow.cat([input_ids, next_tokens[:, None]], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder
            )
            cur_len = cur_len + 1

            if eos_token_id is not None:
                unfinished_sequences = flow.mul(
                    unfinished_sequences, (next_tokens != eos_token_id).long()
                )

            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

        # Release records
        if "past_key_values" in self.__dir__():
            self.past_key_values = [None] * self.cfg.hidden_layers
        if "encoder_states" in self.__dir__():
            self.encoder_states = None

        return input_ids

    def beam_search(
        self,
        input_ids: flow.Tensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        is_encoder_decoder: bool = False,
        output_scores: bool = False,
        **model_kwargs,
    ):
        pad_token_id = pad_token_id if pad_token_id is not None else self.cfg.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id
        output_scores = output_scores if output_scores is not None else self.cfg.output_scores
        scores = () if output_scores else None
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use "
                "`stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))`"
                "instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn(
                "You don't have defined any stopping_criteria, this will likely loop forever",
                UserWarning,
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, "
                f"but is {batch_beam_size}."
            )

        beam_indices = None

        beam_scores = flow.zeros(
            (batch_size, num_beams),
            dtype=flow.float,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs)
            next_token_logits = outputs["logits"][:, -1, :]

            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)
            next_token_scores = next_token_scores.to_global(
                sbp=input_ids.sbp, placement=input_ids.placement
            )

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores
            )

            # Store scores
            if output_scores:
                scores += (next_token_scores,)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = flow.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = flow.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder
            )

            # update past_key_value
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        # Release records
        if "past_key_values" in self.__dir__():
            self.past_key_values = [None] * self.cfg.hidden_layers
        if "encoder_states" in self.__dir__():
            self.encoder_states = None

        return sequence_outputs["sequences"]

    @flow.no_grad()
    def generate(
        self,
        inputs: Optional[flow.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        force_words_ids: Optional[Union[Iterable[int], Iterable[Iterable[int]]]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, flow.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        renormalize_logits: Optional[bool] = None,
        stopping_criteria=StoppingCriteriaList(),
        constraints=None,
        output_scores: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        exponential_decay_length_penalty: Optional[Tuple[Union[int, float]]] = None,
        **model_kwargs,
    ):
        # 0. Validate model kwargs
        self._validate_model_kwargs(model_kwargs.copy())

        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.cfg.bos_token_id
        num_beams = num_beams if num_beams is not None else self.cfg.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.cfg.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.cfg.early_stopping
        num_beam_groups = (
            num_beam_groups if num_beam_groups is not None else self.cfg.num_beam_groups
        )
        do_sample = do_sample if do_sample is not None else self.cfg.do_sample
        num_return_sequences = (
            num_return_sequences
            if num_return_sequences is not None
            else self.cfg.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.cfg.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id

        output_scores = output_scores if output_scores is not None else self.cfg.output_scores

        # 2. Prepare model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 3. Prepare other model kwargs
        model_kwargs["use_cache"] = use_cache if use_cache is not None else self.cfg.use_cache

        if self.cfg.is_encoder_decoder:
            att_mask_name = "encoder_attn_mask"
            accepts_attention_mask = att_mask_name in set(
                inspect.signature(self.forward).parameters.keys()
            )
        else:
            att_mask_name = "attention_mask"
            accepts_attention_mask = att_mask_name in set(
                inspect.signature(self.forward).parameters.keys()
            )
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if (
            model_kwargs.get(att_mask_name, None) is None
            and requires_attention_mask
            and accepts_attention_mask
        ):
            model_kwargs[att_mask_name] = self._prepare_attention_mask_for_generation(
                inputs_tensor, pad_token_id, eos_token_id
            )
        if self.cfg.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.cfg.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        # 5. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        if max_length is None and max_new_tokens is None:
            if dist.is_main_process():
                warnings.warn(
                    "Neither `max_length` nor `max_new_tokens` has been set, `max_length` will "
                    f"default to {self.cfg.max_length} (`self.cfg.max_length`).  we recommend using"
                    " `max_new_tokens` to control the maximum length of the generation.",
                    UserWarning,
                )
        elif max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids_seq_length
        elif max_length is not None and max_new_tokens is not None:
            raise ValueError(
                "Both `max_new_tokens` and `max_length` have been set but they serve the same"
            )

        # default to cfg if still None
        max_length = max_length if max_length is not None else self.cfg.max_length
        min_length = min_length if min_length is not None else self.cfg.min_length

        if min_length is not None and min_length > max_length:
            raise ValueError(
                f"Unfeasable length constraints: the minimum length ({min_length}) is larger than"
                f"the maximum length ({max_length})"
            )
        if input_ids_seq_length >= max_length:
            input_ids_string = "decoder_input_ids" if self.cfg.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is"
                f" set to {max_length}. This can lead to unexpected behavior. You should consider "
                "increasing `max_new_tokens`."
            )

        # 6. Determine generation mode
        is_constraint_gen_mode = constraints is not None or force_words_ids is not None
        is_greedy_gen_mode = (
            (num_beams == 1)
            and (num_beam_groups == 1)
            and do_sample is False
            and not is_constraint_gen_mode
        )
        is_sample_gen_mode = (
            (num_beams == 1)
            and (num_beam_groups == 1)
            and do_sample is True
            and not is_constraint_gen_mode
        )
        is_beam_gen_mode = (
            (num_beams > 1)
            and (num_beam_groups == 1)
            and do_sample is False
            and not is_constraint_gen_mode
        )
        # is_beam_sample_gen_mode = (
        #     (num_beams > 1)
        #     and (num_beam_groups == 1)
        #     and do_sample is True
        #     and not is_constraint_gen_mode
        # )
        is_group_beam_gen_mode = (
            (num_beams > 1) and (num_beam_groups > 1) and not is_constraint_gen_mode
        )

        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is"
                " set to `False`."
            )

        # 7. Prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            logits_processor=logits_processor,
            renormalize_logits=renormalize_logits,
        )

        # 8. Prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length,
            max_time=max_time,
            stopping_criteria=stopping_criteria,
        )

        # 9. Go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing"
                    " greedy search."
                )

            # 10. Run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 10. Prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                temperature=temperature,
                num_beams=num_beams,
                renormalize_logits=renormalize_logits,
            )

            # 11. Expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.cfg.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. Run multinomial sample
            return self.multinomial_sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`."
                )

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 10. Prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )

            # 11. Interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams,
                is_encoder_decoder=self.cfg.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. Run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                **model_kwargs,
            )
