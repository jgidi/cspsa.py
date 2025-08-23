#!/usr/bin/env python3

from functools import partial

from .defaults import DEFAULT_REAL_PERTURBATIONS
from .base_optimizer import CSPSA

SPSA = partial(CSPSA, perturbations=DEFAULT_REAL_PERTURBATIONS)
