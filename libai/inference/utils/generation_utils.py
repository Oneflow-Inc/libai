import warnings
import logging
import inspect
from typing import Callable, List, Optional, Tuple
from copy import deepcopy

import numpy as np
import oneflow as flow
from .generation_beam_search import BeamScorer
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
from oneflow import nn

from libai.utils import distributed as dist


logger = logging.getLogger(__name__)

def multinomial(probs, num_samples, is_global=True):
    probs = probs.numpy()
    res = []
    for i in range(len(probs)):
        target = probs[i]
        samples_idx = []
        for _ in range(num_samples):
            result = np.random.multinomial(1, target, size=1)
            prob = target[result.argmax()]
            samples_idx.append(np.where(probs[i] == prob)[0].tolist()[0])
            target = np.delete(target, result.argmax())
        res.append(samples_idx)
    return (
        flow.tensor(res, sbp=probs.sbp, placement=probs.placement)
        if is_global
        else flow.tensor(res)
    )


def validate_stopping_criteria(stopping_criteria, max_length):
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria


def multinomial_sample(
    model,
    input_ids: flow.Tensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    is_encoder_decoder=False,
    output_scores=False,
    **model_kwargs,
):
    # init values
    scores = () if output_scores else None
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))`"
            "instead.",
            UserWarning,
        )
        stopping_criteria = MaxLengthCriteria(max_length=max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    cur_len = input_ids.shape[-1]

    while True:
        # prepare model inputs
        model_inputs = model._prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = model(**model_inputs)

        next_token_logits = outputs[:, -1, :]
        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores
        if output_scores:
            scores += (next_token_scores,)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = multinomial(
            probs, num_samples=1
        )  # flow.multinomial(probs, num_samples=1).squeeze(1)
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

        model_kwargs = _update_model_kwargs_for_generation(
            model, model_kwargs, is_encoder_decoder=is_encoder_decoder
        )
        cur_len = cur_len + 1

        if eos_token_id is not None:
            unfinished_sequences = flow.mul(
                unfinished_sequences, (next_tokens != eos_token_id).long()
            )

        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break

    return input_ids


def beam_search(
    model,
    input_ids: flow.Tensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    is_encoder_decoder=False,
    output_scores=False,
    **model_kwargs,
):
    scores = () if output_scores else None
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))`"
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
        placement=flow.placement("cuda", list(range(dist.get_world_size()))),
    )
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    while True:
        # prepare model inputs
        model_inputs = model._prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = model(**model_inputs)
        next_token_logits = outputs[:, -1, :]

        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

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

        model_kwargs = _update_model_kwargs_for_generation(
            model, model_kwargs, is_encoder_decoder=is_encoder_decoder
        )

        # update past_key_value
        if model_kwargs["past"] is not None:
            model_kwargs["past"] = model._reorder_cache(beam_idx)

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

    return sequence_outputs["sequences"]


class Generator:
    def _prepare_model_inputs(self, inputs=None, bos_token_id=None, model_kwargs=None):
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
            inputs = self._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))
        return inputs, input_name, model_kwargs

    def prepare_inputs_for_generation(self, input_ids: flow.Tensor, **kwargs):
        """
        Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the
        generate method.
        """
        return {"input_ids": input_ids}

    def _prepare_input_ids_for_generation(self, bos_token_id: int, encoder_outputs):
        if self.cfg.is_encoder_decoder and encoder_outputs is not None:
            shape = encoder_outputs.size()[:-1]
            return (
                flow.ones(
                    shape,
                    dtype=flow.long,
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                    placement=flow.placement("cuda", list(range(dist.get_world_size()))),
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
                placement=flow.placement("cuda", list(range(dist.get_world_size()))),
            )
            * bos_token_id
        )

    def _prepare_attention_mask_for_generation(
        self,
        inputs: flow.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[int],
    ):
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [flow.int64]
        is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs.squeeze(0))
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        # Check if input is input_ids and padded -> only then is attention_mask defined
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return inputs.ne(pad_token_id).long()
        else:
            return flow.ones(
                inputs.shape[:2],
                dtype=flow.long,
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=flow.placement("cuda", list(range(dist.get_world_size()))),
            )

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor, model_kwargs, model_input_name):
        only_encoder = True
        model_kwargs[model_input_name] = inputs_tensor
        if "encoder_decoder_attn_mask" in set(inspect.signature(self.forward).parameters):
            model_kwargs["encoder_decoder_attn_mask"] = model_kwargs["encoder_attn_mask"]
        model_kwargs["encoder_outputs"] = self(**model_kwargs, only_encoder=only_encoder)
        model_kwargs.pop(model_input_name)
        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self, batch_size, decoder_start_token_id, bos_token_id, model_kwargs
    ):
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            return model_kwargs.pop("decoder_input_ids")
        else:
            decoder_start_token_id = decoder_start_token_id if decoder_start_token_id else self.cfg.decoder_start_token_id
            return (
                flow.ones(
                    (batch_size, 1),
                    dtype=flow.long,
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                    placement=flow.placement("cuda", list(range(dist.get_world_size()))),
                )
                * decoder_start_token_id
            )

    def _get_decoder_start_token_id(self, decoder_start_token_id=None, bos_token_id=None):
        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif self.cfg.is_encoder_decoder:
            return self.cfg.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        else:
            return self.cfg.bos_token_idx

    def _update_model_kwargs_for_generation(self, model_kwargs, is_encoder_decoder=False):
        # update past_key_value_state
        if self.past_key_values[-1] is not None:
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
        top_k=None,
        top_p=None,
        typical_p=None,
        temperature=None,
        num_beams=None,
        renormalize_logits=None,
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
        is_encoder_decoder=False,
        output_scores=False,
        **model_kwargs,
    ):
        pad_token_id = pad_token_id if pad_token_id is not None else self.cfg.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id
        output_scores = output_scores if output_scores is not None else self.cfg.output_scores
        scores = () if output_scores else None
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])`"
                " instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)


        # keep track of which sequences are already finished
        unfinished_sequences = flow.zeros(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]
        while True:
            # prepare model inputs
            model_inputs = self._prepare_inputs_for_generation(input_ids, **model_kwargs)
            # generate
            outputs = self(**model_inputs)
            next_token_logits = outputs[:, -1, :]

            # logits_processor
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores
            if output_scores:
                scores += (next_token_scores,)

            # argmax
            next_tokens = flow.argmax(next_token_scores, dim=-1)
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
                model_kwargs, is_encoder_decoder=is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = flow.mul(
                    unfinished_sequences, (next_tokens != eos_token_id).long()
                )

            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

        return input_ids

    @flow.no_grad()
    def generate(
        self,
        inputs=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        typical_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        force_words_ids=None, 
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        encoder_no_repeat_ngram_size=None,
        num_return_sequences=None,
        max_time=None,
        max_new_tokens=None,
        decoder_start_token_id=None,
        use_cache=None,
        num_beam_groups=None,
        diversity_penalty=None,
        prefix_allowed_tokens_fn=None,
        logits_processor=LogitsProcessorList(),
        renormalize_logits=None,
        stopping_criteria=StoppingCriteriaList(),
        constraints=None,
        output_attentions=None,
        output_hidden_states=None,
        output_scores=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        remove_invalid_values=None,
        exponential_decay_length_penalty=None,
        **model_kwargs,
    ):
        # 0. Validate model kwargs
        self._validate_model_kwargs(model_kwargs.copy())

        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.cfg.bos_token_id
        num_beams = num_beams if num_beams is not None else self.cfg.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.cfg.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.cfg.early_stopping
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.cfg.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.cfg.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.cfg.num_return_sequences
        )
        
        pad_token_id = pad_token_id if pad_token_id is not None else self.cfg.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id
        
        output_scores = output_scores if output_scores is not None else self.cfg.output_scores
        
        # 2. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]
        
        # 3. Define other model kwargs
        model_kwargs["use_cache"] = use_cache if use_cache is not None else self.cfg.use_cache
        
        if self.cfg.is_encoder_decoder:
            att_mask_name = "encoder_attn_mask"
            accepts_attention_mask = att_mask_name in set(inspect.signature(self.forward).parameters.keys())
        else:
            att_mask_name = "attention_mask"
            accepts_attention_mask = att_mask_name in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        
        if model_kwargs.get(att_mask_name, None) is None and requires_attention_mask and accepts_attention_mask:
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
            warnings.warn(
                "Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to "
                f"{self.cfg.max_length} (`self.cfg.max_length`).  we recommend using `max_new_tokens` "
                "to control the maximum length of the generation.",
                UserWarning,
            )
        elif max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids_seq_length
        elif max_length is not None and max_new_tokens is not None:
            raise ValueError(
                "Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a"
                " limit to the generated output length. Remove one of those arguments. Please refer to the"
                " documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        # default to cfg if still None
        max_length = max_length if max_length is not None else self.cfg.max_length
        min_length = min_length if min_length is not None else self.cfg.min_length
        
        if min_length is not None and min_length > max_length:
            raise ValueError(
                f"Unfeasable length constraints: the minimum length ({min_length}) is larger than the maximum "
                f"length ({max_length})"
            )
        if input_ids_seq_length >= max_length:
            input_ids_string = "decoder_input_ids" if self.cfg.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {max_length}. This can lead to unexpected behavior. You should consider increasing "
                "`max_new_tokens`."
            )
            
        # 6. determine generation mode
        is_constraint_gen_mode = constraints is not None or force_words_ids is not None
        is_greedy_gen_mode = (
            (num_beams == 1) and (num_beam_groups == 1) and do_sample is False and not is_constraint_gen_mode
        )
        is_sample_gen_mode = (
            (num_beams == 1) and (num_beam_groups == 1) and do_sample is True and not is_constraint_gen_mode
        )
        is_beam_gen_mode = (
            (num_beams > 1) and (num_beam_groups == 1) and do_sample is False and not is_constraint_gen_mode
        )
        is_beam_sample_gen_mode = (
            (num_beams > 1) and (num_beam_groups == 1) and do_sample is True and not is_constraint_gen_mode
        )
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1) and not is_constraint_gen_mode

        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )
        
        # 7. prepare distribution pre_processing samplers
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
        
        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )
        
        # 9. go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # 10. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                **model_kwargs,
            )