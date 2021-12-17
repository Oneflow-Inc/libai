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