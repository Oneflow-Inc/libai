import warnings

from typing import Optional

from .generation_logits_processor import LogitsProcessorList
from .generation_stopping_criteria import MaxLengthCriteria
import oneflow as flow


def _update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
    # update past
    if "past_key_values" in outputs:
        model_kwargs["past"] = outputs.past_key_values

    # update token_type_ids with last value
    if "tokentype_ids" in model_kwargs:
        tokentype_ids = model_kwargs["tokentype_ids"]
        model_kwargs["tokentype_ids"] = flow.cat([tokentype_ids, tokentype_ids[:, -1].unsqueeze(-1)], dim=-1)

    # update attention mask
    if not is_encoder_decoder:
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = flow.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
    return model_kwargs


def greedy_search(
    model,
    input_ids: flow.Tensor, 
    logits_processor: Optional[list] = None, 
    stopping_criteria: Optional[list] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = False,
    output_hidden_states: Optional[bool] = False,
    output_scores: Optional[bool] = False,
    return_dict_in_generate: Optional[bool] = False,
    is_encoder_decoder = False,
    **model_kwargs,
):
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
    
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else MaxLengthCriteria(max_length=max_length)
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = MaxLengthCriteria(max_length=max_length)
    if return_dict_in_generate and is_encoder_decoder:
        encoder_attentions = model_kwargs["attentions"] if output_attentions else None
        encoder_hidden_states = model_kwargs["hidden_states"] if output_hidden_states else None
    
    unfinished_sequences = flow.zeros(input_ids.shape[0]).fill_(1)
    cur_len = input_ids.shape[-1]
    while True:
        # prepare model inputs
        model_inputs = {"input_ids": input_ids, **model_kwargs}
        outputs = model(
            **model_inputs
        )
        next_token_logits = outputs[:, -1, :]

        # logits_processor
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # store scores, attention, hidden_state
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores, )
            if output_attentions:
                decoder_attentions += (outputs["decoder_attentions"],) if is_encoder_decoder else (outputs["attentions"],)
                if is_encoder_decoder:
                    cross_attentions += outputs["cross_attentions"]
            if output_hidden_states:
                decoder_hidden_states += (outputs["decoder_hidden_states"],) if is_encoder_decoder else (outputs["hidden_states"],)

        # argmax
        next_tokens = flow.argmax(next_tokens_scores, dim=-1)
        unfinished_sequences = unfinished_sequences.to_global(sbp=next_tokens.sbp, placement=next_tokens.placement)

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = flow.cat([input_ids, next_tokens[:, None]], dim=-1)
        # model_kwargs = _update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder = is_encoder_decoder)
        cur_len = cur_len + 1

        if eos_token_id is not None:
            unfinished_sequences = flow.mul(unfinished_sequences, (next_tokens != eos_token_id).long())

        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break

    if return_dict_in_generate:
        if is_encoder_decoder:
            return (
                input_ids,
                scores,
                encoder_attentions,
                encoder_hidden_states,
                decoder_attentions,
                cross_attentions,
                decoder_hidden_states
            )
        else:
            return (
                input_ids,
                scores,
                decoder_attentions,
                decoder_hidden_states
            )
    else:
        return input_ids  