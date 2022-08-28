from curses import noecho
import warnings

from typing import Optional

from .generation_stopping_criteria import MaxLengthCriteria
import oneflow as flow


# def max_length_criteria(input_ids, max_length):
    

# def validate_stopping_criteria(stopping_criteria, max_length: int):
#     stopping_max_length = stopping_criteria.max_length
#     new_stopping_criteria = deepcopy(stopping_criteria)
#     if stopping_max_length is not None and stopping_max_length != max_length:
#         warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
#     elif stopping_max_length is None:
#         new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
#     return new_stopping_criteria


def greedy_search(
    model,
    input_ids: flow.LongTensor, 
    logits_processor: Optional[list] = None, 
    stopping_criteria: Optional[list] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    is_encoder_decoder = False,
    **model_kwargs,
):
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
    
    logits_processor = logits_processor if logits_processor is not None else []
    stopping_criteria = stopping_criteria if stopping_criteria is not None else []
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
        model_inputs = {"input_ids": input_ids}
        outputs = model(
            **model_inputs
        )
        next_token_logits = outputs["logits"][:, -1, :]
        
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
        
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
        input_ids = flow.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = _update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder = is_encoder_decoder)
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