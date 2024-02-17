#!/usr/bin/env python3

import numpy as np
from copy import copy
from tqdm import tqdm
from collections.abc import Callable, Iterable

from .defaults import *

class CSPSA:
    def __init__(self,
                 num_iter : int = DEFAULT_NUM_ITER,
                 gains : dict = DEFAULT_GAINS,
                 init_iter : int = 0,
                 callback : Callable = None,
                 perturbations = DEFAULT_COMPLEX_PERTURBATIONS,
                 ):

        self.gains = copy(gains)
        self.num_iter = num_iter
        self.init_iter = init_iter
        self.iter = self.init_iter
        self.pert = perturbations
        self.callback = callback

    def restart(self):
        self.iter = self.init_iter

    def _stepsize_and_pert(self):
        a = self.gains.get('a', DEFAULT_GAINS['a'])
        A = self.gains.get('A', DEFAULT_GAINS['A'])
        b = self.gains.get('b', DEFAULT_GAINS['b'])
        s = self.gains.get('s', DEFAULT_GAINS['s'])
        t = self.gains.get('t', DEFAULT_GAINS['t'])

        ak = a / (self.iter + 1 + A)**s
        bk = b / (self.iter + 1)**t

        return ak, bk

    def step(self, fun, guess):
        if self.iter >= self.init_iter + self.num_iter:
            raise Exception("Maximum number of iterations achieved")

        ak, bk = self._stepsize_and_pert()

        delta = bk * np.random.choice(self.pert, len(guess))

        df = fun(guess + delta) - fun(guess - delta)
        grad = 0.5 * df / np.conj(delta)

        new_guess = guess - ak * grad

        if self.callback is not None:
            self.callback(self.iter, new_guess)

        self.iter += 1

        return new_guess

    def run(self, fun : Callable, guess : Iterable,
            progressbar : bool = False) -> np.ndarray:
        new_guess = np.copy(guess)

        iterator = range(self.init_iter, self.init_iter + self.num_iter)
        for _ in tqdm(iterator, disable=not progressbar):
            new_guess = self.step(fun, new_guess)

        return new_guess
