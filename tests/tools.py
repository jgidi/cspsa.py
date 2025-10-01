#!/usr/bin/env python3

import numpy as np
from cspsa.defaults import (
    DEFAULT_NUM_ITER,
    DEFAULT_COMPLEX_PERTURBATIONS,
    DEFAULT_GAINS,
)


def naive_first_order(
    f,
    guess,
    num_iter=DEFAULT_NUM_ITER,
    perturbations=DEFAULT_COMPLEX_PERTURBATIONS,
    gains=DEFAULT_GAINS,
    accumulate=False,
    rng=None,
):
    params = np.copy(guess)
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None")
    if rng is None:
        rng = np.random.default_rng()
    else:
        rng_state = rng.__getstate__()
        rng = np.random.default_rng()
        rng.__setstate__(rng_state)  # Copy the state

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
