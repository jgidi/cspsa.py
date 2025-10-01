#!/usr/bin/env python3

import numpy as np
import unittest

from cspsa import SPSA, CSPSA
from cspsa.defaults import (
    DEFAULT_REAL_PERTURBATIONS,
    DEFAULT_COMPLEX_PERTURBATIONS,
)

from .tools import naive_first_order


def complex_objfun(x):
    return np.abs(x[0] - 1 - 3j) + np.abs(x[1])


def real_objfun(x):
    return (x[0] - 4) ** 2 + (x[1] - 0.5) ** 2


class FirstOrder(unittest.TestCase):
    def test_spsa(self):
        rng = np.random.default_rng(0)
        objfun = real_objfun
        guess = np.random.randn(2)

        opt = SPSA(rng=rng)
        params1 = opt.make_params_collector()
        opt.run(objfun, guess)

        params2 = naive_first_order(
            objfun,
            guess,
            accumulate=True,
            perturbations=DEFAULT_REAL_PERTURBATIONS,
            rng=rng,
        )

        for x1, x2 in zip(params1, params2):
            self.assertTrue(np.all(x1 == x2))

    def test_cspsa(self):
        rng = np.random.default_rng(0)
        objfun = complex_objfun
        guess = np.random.randn(2)

        opt = CSPSA(rng=rng)
        params1 = opt.make_params_collector()
        opt.run(objfun, guess)

        np.random.seed(0)
        params2 = naive_first_order(
            objfun,
            guess,
            accumulate=True,
            perturbations=DEFAULT_COMPLEX_PERTURBATIONS,
            rng=rng,
        )

        for x1, x2 in zip(params1, params2):
            self.assertTrue(np.allclose(x1, x2))


if __name__ == "__main__":
    unittest.main()
