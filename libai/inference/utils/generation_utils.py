import warnings

from typing import Optional

from .generation_logits_processor import LogitsProcessorList
from .generation_stopping_criteria import MaxLengthCriteria, StoppingCriteriaList
import oneflow as flow


def _update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
    # update attention mask
    if "attention_mask" in model_kwargs and not is_encoder_decoder:
        attention_mask = model_kwargs["attention_mask"]
        pad = flow.ones((attention_mask.shape[0], 1), sbp = attention_mask.sbp, placement=attention_mask.placement)
        model_kwargs["attention_mask"] = flow.cat([attention_mask, pad], dim=-1)
        
    if "decoder_attn_mask" in model_kwargs and is_encoder_decoder:
        attention_mask = model_kwargs["decoder_attn_mask"]
        pad = flow.ones((attention_mask.shape[0], 1), sbp = attention_mask.sbp, placement=attention_mask.placement)
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
    is_encoder_decoder = False,
    output_scores=False,
    **model_kwargs,
):
    scores = () if output_scores else None
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
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
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        
        # Store scores
        if output_scores:
            scores += (next_tokens_scores,)

        # argmax
        next_tokens = flow.argmax(next_tokens_scores, dim=-1)
        unfinished_sequences = unfinished_sequences.to_global(sbp=next_tokens.sbp, placement=next_tokens.placement)

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        next_tokens = next_tokens.to(flow.long)
        if is_encoder_decoder:
            model_kwargs["decoder_input_ids"] = flow.cat([model_kwargs["decoder_input_ids"], next_tokens[:, None]], dim=-1)
        else:    
            input_ids = flow.cat([input_ids, next_tokens[:, None]], dim=-1)

        model_kwargs = _update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder = is_encoder_decoder)
        cur_len = cur_len + 1

        if eos_token_id is not None:
            unfinished_sequences = flow.mul(unfinished_sequences, (next_tokens != eos_token_id).long())

        if is_encoder_decoder:
            if unfinished_sequences.max() == 0 or stopping_criteria(model_kwargs["decoder_input_ids"], scores):
                break
        else:
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

    if is_encoder_decoder:
        return model_kwargs["decoder_input_ids"]
    return input_ids


def sample(
    model,
    input_ids: flow.Tensor, 
    logits_processor: Optional[LogitsProcessorList] = None, 
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    is_encoder_decoder = False,
    output_scores=False,
    **model_kwargs,
):
    scores = () if output_scores else None
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
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
