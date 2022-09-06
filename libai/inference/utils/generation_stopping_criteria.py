import oneflow as flow


class StoppingCriteriaList(list):
    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor, **kwargs) -> bool:
        return any(criteria(input_ids, scores) for criteria in self)

    @property
    def max_length(self):
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
        return None


class MaxLengthCriteria(object):
    def __init__(self, max_length: int):
        self.max_length = max_length
    
    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor):
        return input_ids.shape[-1] >= self.max_length