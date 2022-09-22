import warnings
from typing import Optional

import numpy as np
import oneflow as flow
from generation_beam_search import BeamScorer
from generation_logits_processor import LogitsProcessorList
from generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from oneflow import nn

from libai.utils import distributed as dist


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


def _update_model_kwargs_for_generation(model, model_kwargs, is_encoder_decoder=False):
    # update past_key_value_state
    if model.past_key_values[-1] is not None:
        model_kwargs["past"] = model.past_key_values
    else:
        model_kwargs["past"] = None

    # update attention mask
    if "attention_mask" in model_kwargs and not is_encoder_decoder:
        attention_mask = model_kwargs["attention_mask"]
        pad = flow.ones(
            (attention_mask.shape[0], 1), sbp=attention_mask.sbp, placement=attention_mask.placement
        )
        model_kwargs["attention_mask"] = flow.cat([attention_mask, pad], dim=-1)

    if "decoder_attn_mask" in model_kwargs and is_encoder_decoder:
        attention_mask = model_kwargs["decoder_attn_mask"]
        pad = flow.ones(
            (attention_mask.shape[0], 1), sbp=attention_mask.sbp, placement=attention_mask.placement
        )
        model_kwargs["decoder_attn_mask"] = flow.cat([attention_mask, pad], dim=-1)

    return model_kwargs


def greedy_search(
    model,
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
    scores = () if output_scores else None
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])`"
            " instead.",
            UserWarning,
        )
        stopping_criteria = MaxLengthCriteria(max_length=max_length)

    if is_encoder_decoder:
        unfinished_sequences = flow.zeros(model_kwargs["decoder_input_ids"].shape[0]).fill_(1)
        cur_len = model_kwargs["decoder_input_ids"].shape[-1]
    else:
        unfinished_sequences = flow.zeros(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

    while True:
        # prepare model inputs
        if is_encoder_decoder:
            model_inputs = {"encoder_input_ids": input_ids, **model_kwargs}
        else:
            model_inputs = {"input_ids": input_ids, **model_kwargs}

        outputs = model(**model_inputs)

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
        if is_encoder_decoder:
            model_kwargs["decoder_input_ids"] = flow.cat(
                [model_kwargs["decoder_input_ids"], next_tokens[:, None]], dim=-1
            )
        else:
            input_ids = flow.cat([input_ids, next_tokens[:, None]], dim=-1)

        model_kwargs = _update_model_kwargs_for_generation(
            model, model_kwargs, is_encoder_decoder=is_encoder_decoder
        )
        cur_len = cur_len + 1

        if eos_token_id is not None:
            unfinished_sequences = flow.mul(
                unfinished_sequences, (next_tokens != eos_token_id).long()
            )

        if is_encoder_decoder:
            if unfinished_sequences.max() == 0 or stopping_criteria(
                model_kwargs["decoder_input_ids"], scores
            ):
                break
        else:
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

    if is_encoder_decoder:
        return model_kwargs["decoder_input_ids"]
    return input_ids


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

    if is_encoder_decoder:
        unfinished_sequences = flow.zeros(model_kwargs["decoder_input_ids"].shape[0]).fill_(1)
        cur_len = model_kwargs["decoder_input_ids"].shape[-1]
    else:
        unfinished_sequences = flow.zeros(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

    while True:
        # prepare model inputs
        if is_encoder_decoder:
            model_inputs = {"encoder_input_ids": input_ids, **model_kwargs}
        else:
            model_inputs = {"input_ids": input_ids, **model_kwargs}

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
        if is_encoder_decoder:
            model_kwargs["decoder_input_ids"] = flow.cat(
                [model_kwargs["decoder_input_ids"], next_tokens[:, None]], dim=-1
            )
        else:
            input_ids = flow.cat([input_ids, next_tokens[:, None]], dim=-1)

        model_kwargs = _update_model_kwargs_for_generation(
            model, model_kwargs, is_encoder_decoder=is_encoder_decoder
        )
        cur_len = cur_len + 1

        if eos_token_id is not None:
            unfinished_sequences = flow.mul(
                unfinished_sequences, (next_tokens != eos_token_id).long()
            )

        if is_encoder_decoder:
            if unfinished_sequences.max() == 0 or stopping_criteria(
                model_kwargs["decoder_input_ids"], scores
            ):
                break
        else:
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

    if is_encoder_decoder:
        return model_kwargs["decoder_input_ids"]
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
