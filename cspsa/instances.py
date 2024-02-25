#!/usr/bin/env python3

from functools import partial

from .stochastic_optimizer import StochasticOptimizer
from .defaults import *

SPSA = partial(StochasticOptimizer, perturbations=DEFAULT_REAL_PERTURBATIONS)
SPSA.__doc__ = """
My docs
"""


CSPSA = partial(StochasticOptimizer, perturbations=DEFAULT_COMPLEX_PERTURBATIONS)
CSPSA.__doc__ = """
My docs
"""
