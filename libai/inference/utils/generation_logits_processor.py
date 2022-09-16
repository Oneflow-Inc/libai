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
        scores = flow.scatter(scores, 1, input_ids, score)
        return scores


class TemperatureLogitsWarper(object):
    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")
        self.temperature = temperature
        
    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor):
        scores = scores / self.temperature
        return scores


class TopPLogitsWarper(object):
    # 删除累加概率中大于top_p的概率
    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor):
        sorted_logits, sorted_indices = flow.sort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = flow.scatter(sorted_indices_to_remove, 1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopKLogitsWarper(object):
    # 删除所有小于topk中最小概率的概率
    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor):
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < flow.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
    
    
class TypicalLogitsWarper(object):
    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        mass = float(mass)
        if not (mass > 0 and mass < 1):
            raise ValueError(f"`typical_p` has to be a float > 0 and < 1, but is {mass}")

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor):

        # calculate entropy
        normalized = flow.nn.functional.log_softmax(scores, dim=-1)
        p = flow.exp(normalized)
        x = normalized * p
        shape = list(x.size())
        shape[-1] = 1
        x[flow.where(x != x)] = 0
        ent = -x.sum(dim=-1).reshape(shape)

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
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = flow.scatter(sorted_indices_to_remove, 1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
