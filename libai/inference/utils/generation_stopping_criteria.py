import time
import warnings
from copy import deepcopy

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


class MaxTimeCriteria(object):
    def __init__(self, max_time: float, initial_timestamp: float = None):
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor, **kwargs):
        return time.time() - self.initial_timestamp > self.max_time


def validate_stopping_criteria(
    stopping_criteria: StoppingCriteriaList, max_length: int
) -> StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn(
            "You set different `max_length` for stopping criteria and `max_length` parameter",
            UserWarning,
        )
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria
