"""
Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/solver/lr_scheduler.py
"""

import logging
import math
from bisect import bisect_right
from typing import List
import oneflow as flow

from .param_scheduler import ParamScheduler

logger = logging.getLogger(__name__)


class LRMultiplier(flow.optim.lr_scheduler._LRScheduler):
    """
    A LRScheduler which uses :class:`ParamScheduler` to multiply the
    learning rate of each param in the optimizer.
    Every step, the learning rate of each parameter becomes its initial value
    multiplied by the output of the given :class:`ParamScheduler`.

    The absolute learning rate value of each parameter can be different.
    This scheduler can be used as long as the relative scale among them do
    not change during training.
    
    Examples:
    ::
        LRMultiplier(
            opt,
            WarmupParamScheduler(
                MultiStepParamScheduler(
                    [1, 0.1, 0.01],
                    milestones=[60000, 80000],
                    num_updates=90000,
                ), 0.001, 100 / 90000
            ),
            max_iter=90000
        )
    """

    # NOTES: in the most general case, every LR can use its own scheduler.
    # Supporting this requires interaction with the optimizer when its parameter
    # group is initialized. For example, classyvision implements its own optimizer
    # that allows different schedulers for every parameter group.
    # To avoid this complexity, we use this class to support the most common cases
    # where the relative scale among all LRs stay unchanged during training.  In this
    # case we only need a total of one scheduler that defines the relative LR multiplier.

    def __init__(
        self,
        optimizer: flow.optim.Optimizer,
        multiplier: ParamScheduler,
        max_iter: int,
        last_iter: int = -1,
    ):
        """
        Args:
            optimizer, last_iter: See ``flow.optim.lr_scheduler._LRScheduler``.
                ``last_iter`` is the same as ``last_epoch``.
            multiplier: a libai ParamScheduler that defines the multiplier on
                every LR of the optimizer
            max_iter: the total number of training iterations
        """
        if not isinstance(multiplier, ParamScheduler):
            raise ValueError(
                "_LRMultiplier(multiplier=) must be an instance of libai "
                f"ParamScheduler. Got {multiplier} instead."
            )
        self._multiplier = multiplier
        self._max_iter = max_iter
        super().__init__(optimizer, last_step=last_iter)
    
    def state_dict(self):
        # libai schedulers are stateless. Only keep oneflow scheduler states
        return {"base_lrs": self.base_lrs, "last_step": self.last_step}
    
    def get_lr(self) -> List[float]:
        multiplier = self._multiplier(self.last_step / self._max_iter)
        return [base_lr * multiplier for base_lr in self.base_lrs]