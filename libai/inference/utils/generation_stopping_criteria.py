import oneflow as flow


class MaxLengthCriteria(object):
    def __init__(self, max_length: int):
        self.max_length = max_length
    
    def __call__(self, input_ids: flow.Tensor, scores: flow.Tensor):
        return input_ids.shape[-1] >= self.max_length