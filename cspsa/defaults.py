#!/usr/bin/env python3

DEFAULT_NUM_ITER = 300

# Gain parameters
STANDARD_GAINS = {
    "a": 3.0,
    "b": 0.1,
    "A": 0.0,
    "s": 0.602,
    "t": 0.101,
}

ASYMPTOTIC_GAINS = {
    "a": 3.0,
    "b": 0.1,
    "A": 1.0,
    "s": 1.0,
    "t": 1 / 6,
}

DEFAULT_GAINS = STANDARD_GAINS

# Perturbations
DEFAULT_REAL_PERTURBATIONS = (-1, 1)
DEFAULT_COMPLEX_PERTURBATIONS = (-1, -1j, 1, 1j)

# Hessian postprocessing
DEFAULT_HESSIAN_POSTPROCESS_METHOD = "Gidi"
DEFAULT_HESSIAN_POSTPROCESS_TOL = 1e-3
