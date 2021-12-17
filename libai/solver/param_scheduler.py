"""
Modified from https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/param_scheduler.py
"""

import bisect
import math
from typing import List, Optional, Sequence, Union


__all__ = [
    "ParamScheduler",
    "ConstantParamScheduler",
    "CosineParamScheduler",
    "ExponentialParamScheduler",
    "LinearParamScheduler",
    "CompositeParamScheduler",
    "MultiStepParamScheduler",
    "StepParamScheduler",
    "StepWithFixedGammaParamScheduler",
    "PolynomialDecayParamScheduler",
]


class ParamScheduler:
    """
    Base class for parameter schedulers.
    A parameter scheduler defines a mapping from a progress value in [0, 1) to
    a number (e.g. learning rate).
    """

    # To be used for comparisons with where
    WHERE_EPSILON = 1e-6

    def __call__(self, where: float) -> float:
        """
        Get the value of the param for a given point at training.
        We update params (such as learning rate) based on the percent progress
        of training completed. This allows a scheduler to be agnostic to the
        exact length of a particular run (e.g. 120 epochs vs 90 epochs), as
        long as the relative progress where params should be updated is the same.
        However, it assumes that the total length of training is known.
        Args:
            where: A float in [0,1) that represents how far training has progressed
        """
        raise NotImplementedError("Param schedulers must override __call__")
    

class ConstantParamScheduler(ParamScheduler):
    """
    Returns a constant value for a param
    """

    def __init__(self, value: float) -> None:
        self._value = value
    
    def __call__(self, where: float) -> float:
        if where >= 1.0:
            raise RuntimeError(
                f"where in ParamScheduler must be in [0, 1]: got {where}"
            )
        return self._value


class CosineParamScheduler(ParamScheduler):
    """
    Cosine decay or cosine warmup schedules based on start and end values.
    The schedule is updated based on the fraction of training progress.
    The schedule was proposed in 'SGDR: Stochastic Gradient Descent with
    Warm Restarts' (https://arxiv.org/abs/1608.03983). Note that this class
    only implements the cosine annealing part of SGDR, and not the restarts.

    Example:

        .. code-block:: python

          CosineParamScheduler(start_value=0.1, end_value=0.0001)
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
    ) -> None:
        self._start_value = start_value
        self._end_value = end_value

    def __call__(self, where: float) -> float:
        return self._end_value + 0.5 * (self._start_value - self._end_value) * (
            1 + math.cos(math.pi * where)
        )


class ExponentialParamScheduler(ParamScheduler):
    """
    Exponetial schedule parameterized by a start value and decay.
    The schedule is updated based on the fraction of training
    progress, `where`, with the formula
    `param_t = start_value * (decay ** where)`.

    Example:

        .. code-block:: python

            ExponentialParamScheduler(start_value=2.0, decay=0.02)

    Corresponds to a decreasing schedule with values in [2.0, 0.04).
    """

    def __init__(
        self,
        start_value: float,
        decay: float,
    ) -> None:
        self._start_value = start_value
        self._decay = decay

    def __call__(self, where: float) -> float:
        return self._start_value * (self._decay ** where)


class LinearParamScheduler(ParamScheduler):
    """
    Linearly interpolates parameter between ``start_value`` and ``end_value``.
    Can be used for either warmup or decay based on start and end values.
    The schedule is updated after every train step by default.

    Example:

        .. code-block:: python

            LinearParamScheduler(start_value=0.0001, end_value=0.01)

    Corresponds to a linear increasing schedule with values in [0.0001, 0.01)
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
    ) -> None:
        self._start_value = start_value
        self._end_value = end_value

    def __call__(self, where: float) -> float:
        # interpolate between start and end values
        return self._end_value * where + self._start_value * (1 - where)


class MultiStepParamScheduler(ParamScheduler):
    """
    Takes a predefined schedule for a param value, and a list of epochs or steps
    which stand for the upper boundary (excluded) of each range.

    Example:

        .. code-block:: python

          MultiStepParamScheduler(
            values=[0.1, 0.01, 0.001, 0.0001],
            milestones=[30, 60, 80, 120]
          )

    Then the param value will be 0.1 for epochs 0-29, 0.01 for
    epochs 30-59, 0.001 for epochs 60-79, 0.0001 for epochs 80-120.
    Note that the length of values must be equal to the length of milestones
    plus one.
    """

    def __init__(
        self,
        values: List[float],
        num_updates: Optional[int] = None,
        milestones: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            values: param value in each range
            num_updates: the end of the last range. If None, will use ``milestones[-1]``
            milestones: the boundary of each range. If None, will evenly split ``num_updates``
        
        For example, all the following combinations define the same scheduler:

        * num_updates=90, milestones=[30, 60], values=[1, 0.1, 0.01]
        * num_updates=90, values=[1, 0.1, 0.01]
        * milestones=[30, 60, 90], values=[1, 0.1, 0.01]
        * milestones=[3, 6, 9], values=[1, 0.1, 0.01]  (ParamScheduler is scale-invariant)
        """
        if num_updates is None and milestones is None:
            raise ValueError("num_updates and milestones cannot both be None")
        if milestones is None:
            # Default equispaced drop_epochs behavior
            milestones = []
            step_width = math.ceil(num_updates / float(len(values)))
            for idx in range(len(values) - 1):
                milestones.append(step_width * (idx + 1))
        else:
            if not (
                isinstance(milestones, Sequence)
                and len(milestones) == len(values) - int(num_updates is not None)
            ):
                raise ValueError(
                    "MultiStep scheduler requires a list of %d miletones"
                    % (len(values) - int(num_updates is not None))
                )

        if num_updates is None:
            num_updates, milestones = milestones[-1], milestones[:-1]
        if num_updates < len(values):
            raise ValueError(
                "Total num_updates must be greater than length of param schedule"
            )

        self._param_schedule = values
        self._num_updates = num_updates
        self._milestones: List[int] = milestones

        start_epoch = 0
        for milestone in self._milestones:
            # Do not exceed the total number of epochs
            if milestone >= self._num_updates:
                raise ValueError(
                    "Milestone must be smaller than total number of updates: "
                    "num_updates=%d, milestone=%d" % (self._num_updates, milestone)
                )
            # Must be in ascending order
            if start_epoch >= milestone:
                raise ValueError(
                    "Milestone must be smaller than start epoch: start_epoch=%d, milestone=%d"
                    % (start_epoch, milestone)
                )
            start_epoch = milestone

    def __call__(self, where: float) -> float:
        if where > 1.0:
            raise RuntimeError(
                f"where in ParamScheduler must be in [0, 1]: got {where}"
            )
        epoch_num = int((where + self.WHERE_EPSILON) * self._num_updates)
        return self._param_schedule[bisect.bisect_right(self._milestones, epoch_num)]


class PolynomialDecayParamScheduler(ParamScheduler):
    """
    Decays the param value after every epoch according to a
    polynomial function with a fixed power.
    The schedule is updated after every train step by default.

    Example:

        .. code-block:: python

          PolynomialDecayParamScheduler(base_value=0.1, power=0.9)
          
    Then the param value will be 0.1 for epoch 0, 0.099 for epoch 1, and
    so on.
    """

    def __init__(
        self,
        base_value: float,
        power: float,
    ) -> None:
        self._base_value = base_value
        self._power = power

    def __call__(self, where: float) -> float:
        return self._base_value * (1 - where) ** self._power