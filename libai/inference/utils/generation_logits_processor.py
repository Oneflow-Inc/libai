import inspect

import oneflow as flow


class LogitsProcessorList(list):
    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor, **kwargs):
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


class NormalizationLogitsProcessor(object):
    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor):
        scores = scores.log_softmax(dim=-1)
        return scores


class InfNanRemoveLogitsProcessor(object):
    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor):
        scores[scores != scores] = 0.0
        scores[scores == float("inf")] = 1e6
        return scores


class ForcedEOSTokenLogitsProcessor(object):
    def __init__(self, max_length: int, eos_token_id: int):
        self.max_length = max_length
        self.eos_token_id = eos_token_id
    
    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor):
        cur_len = input_ids.shape[-1]
        if cur_len == self.max_length - 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]] = -float("inf")
            scores[:, self.eos_token_id] = 0
        return scores


class ForcedBOSTokenLogitsProcessor(object):
    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id
        
    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor):
        cur_len = input_ids.shape[-1]
        if cur_len == 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.bos_token_id]] = -float("inf")
            scores[:, self.bos_token_id] = 0
        return scores


class TopKLogitsProcessor(object):
    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor):
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))
        index_to_remove = scores < flow.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(index_to_remove, self.filter_value)
        return scores


class RepetitionPenaltyLogitsProcessor(object):
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")
        self.penalty = penalty
    
    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor):
        score = flow.gather(scores, 1, input_ids)
        score = flow.where(score < 0, score * self.penalty, score / self.penalty)

    
class TemperatureLogitsWarper(object):
    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")
        self.temperature = temperature
        
    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor):
        scores = scores / self.temperature
        return scores

