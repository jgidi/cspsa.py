#!/usr/bin/env python3

import numpy as np
from cspsa.defaults import *


def naive_first_order(
    f,
    guess,
    num_iter=DEFAULT_NUM_ITER,
    perturbations=DEFAULT_COMPLEX_PERTURBATIONS,
    gains=DEFAULT_GAINS,
    accumulate=False,
    seed=None,
):
    params = np.copy(guess)
    rng = np.random.default_rng(seed)

    if accumulate:
        acc = []

    for k in range(num_iter):
        ak = gains["a"] / (k + gains["A"] + 1) ** gains["s"]
        bk = gains["b"] / (k + 1) ** gains["t"]

        delta = bk * rng.choice(perturbations, params.shape)

        df = f(params + delta) - f(params - delta)

        params = params - 0.5 * ak * df / delta.conj()

        if accumulate:
            acc.append(params)

    if accumulate:
        return acc
    else:
        return params
