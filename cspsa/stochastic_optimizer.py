#!/usr/bin/env python3

import numpy as np
from scipy import linalg as la

from copy import copy
from tqdm import tqdm
from typing import Union, Callable, Sequence

from .defaults import *


class StochasticOptimizer:
    def __init__(
        self,
        gains: dict = DEFAULT_GAINS,
        init_iter: int = 0,
        callback: Callable = lambda self, x: None,
        postprocessing: Callable = lambda x: x,
        perturbations: Sequence = DEFAULT_REAL_PERTURBATIONS,
        scalar: bool = False,
        second_order: bool = False,
        quantum_natural: bool = False,
        hessian_postprocess_method: str = DEFAULT_HESSIAN_POSTPROCESS_METHOD,
    ):

        self.gains = copy(gains)
        self.callback = callback
        self.init_iter = init_iter
        self.perturbations = perturbations
        self.postprocessing = postprocessing

        # Preconditioning
        self.scalar = scalar
        self.second_order = second_order
        self.quantum_natural = quantum_natural
        self.hessian_postprocess_method = hessian_postprocess_method

        if self.second_order or self.quantum_natural:
            self.compute_update = preconditioned_update
        else:
            self.compute_update = first_order_update

        self.restart()
        self._check_args()

    @property
    def iter_count(self):
        return self.iter - self.init_iter

    def restart(self):
        self.stop = False
        self.iter = self.init_iter
        self.function_eval_count = 0
        self.fidelity_eval_count = 0

    # TODO
    def _check_args(self):
        errmsg = "Can't set both 'second_order=True' and 'quantum_natural=True'"
        assert not (self.second_order and self.quantum_natural), errmsg

        errmsg = "Can't set 'scalar=True' if not using second_order or quantum_natural"
        preconditioned = self.second_order or self.quantum_natural
        assert not (self.scalar and (not preconditioned)), errmsg

    def _stepsize_and_pert(self):
        a = self.gains.get("a", DEFAULT_GAINS["a"])
        A = self.gains.get("A", DEFAULT_GAINS["A"])
        b = self.gains.get("b", DEFAULT_GAINS["b"])
        s = self.gains.get("s", DEFAULT_GAINS["s"])
        t = self.gains.get("t", DEFAULT_GAINS["t"])

        if self.second_order or self.quantum_natural:
            a = 1

        ak = a / (self.iter + 1 + A) ** s
        bk = b / (self.iter + 1) ** t

        return ak, bk

    def default_hessian(self, guess, hessian = None):
        # If provided, return it identically
        if hessian is not None:
            return hessian

        # If not provided, return identity with the right size
        if self.scalar:
            hessian = 1.0
        else:
            hessian = np.eye(len(guess))

        return hessian

    
    def step(
        self,
        fun: Callable,
        guess: np.ndarray,
        previous_hessian: Union[np.ndarray, None] = None,
        fidelity: Union[Callable, None] = None,
    ):

        preconditioned = self.second_order or self.quantum_natural

        if preconditioned:
            previous_hessian = self.default_hessian(guess, previous_hessian)

            update, hessian = preconditioned_update(
                self, fun, guess, previous_hessian, fidelity
            )
        else:
            update = first_order_update(self, fun, guess)

        new_guess = guess - update
        new_guess = self.postprocessing(new_guess)

        self.callback(self, new_guess)
        self.iter += 1

        if preconditioned:
            return new_guess, hessian
        else:
            return new_guess

    def run(
        self,
        fun: Callable,
        guess: Sequence,
        num_iter: int = DEFAULT_NUM_ITER,
        progressbar: bool = False,
        initial_hessian=None,
        fidelity=None,
    ) -> np.ndarray:

        new_guess = np.copy(guess)

        # Preconditioning
        preconditioned = self.second_order or self.quantum_natural
        if preconditioned:
            H = self.default_hessian(guess, initial_hessian)

        iterator = range(self.init_iter, self.init_iter + num_iter)
        for _ in tqdm(iterator, disable=not progressbar):
            if preconditioned:
                new_guess, H = self.step(fun, new_guess, H, fidelity)
            else:
                new_guess = self.step(fun, new_guess)

            if self.stop:
                break

        return new_guess


def scalar_hessian_postprocess(
    self: "StochasticOptimizer",
    Hprev: float,
    H: float,
    method: str = DEFAULT_HESSIAN_POSTPROCESS_METHOD,
    tol: float = DEFAULT_HESSIAN_POSTPROCESS_TOL,
):

    k = self.iter
    if method == "Gidi":
        H = np.sqrt(H**2 + tol)
        H = (k * Hprev + H) / (k + 1)
    else:
        H = (k * Hprev + H) / (k + 1)
        H = np.abs(H) + tol

    return H


def hessian_postprocess(
    self: "StochasticOptimizer",
    H_old: np.ndarray,
    H: np.ndarray,
    method: str = DEFAULT_HESSIAN_POSTPROCESS_METHOD,
    tol: float = DEFAULT_HESSIAN_POSTPROCESS_TOL,
):

    k = self.iter
    I = np.eye(H.shape[0])

    H = (H + H.T.conj()) / 2
    if method == "Gidi":
        H = la.sqrtm(H @ H.T.conj() + tol * I)
        H = (k * H_old + H) / (k + 1)
    else:
        H = (k * H_old + H) / (k + 1)
        H = la.sqrtm(H @ H.T.conj()) + tol * I

    return H


def first_order_update(self: "StochasticOptimizer", fun, guess):

    ak, bk = self._stepsize_and_pert()

    delta = bk * np.random.choice(self.perturbations, len(guess))
    df = fun(guess + delta) - fun(guess - delta)

    self.function_eval_count += 2

    g = 0.5 * ak * df / delta.conj()

    return g


def preconditioned_update(
    self: "StochasticOptimizer",
    fun: Callable,
    guess: np.ndarray,
    previous_hessian: Union[np.ndarray, float, None] = None,
    fidelity: Union[Callable, None] = None,
):

    ak, bk = self._stepsize_and_pert()

    delta = bk * np.random.choice(self.perturbations, len(guess))
    delta2 = bk * np.random.choice(self.perturbations, len(guess))

    # First order
    df = fun(guess + delta) - fun(guess - delta)
    self.function_eval_count += 2

    # Gradient estimator
    g = 0.5 * df / delta.conj()

    # Second order
    if self.second_order:
        dfp = fun(guess + delta + delta2) - fun(guess - delta + delta2)

        self.function_eval_count += 2

        # Hessian factor
        h = 0.5 * (dfp - df)

    # Quantum Natural
    if self.quantum_natural:
        errmsg = "For Quantum Natural optimization, you must provide the fidelity"
        assert fidelity is not None, errmsg

        dF = (
            fidelity(guess, guess + delta + delta2)
            - fidelity(guess, guess - delta + delta2)
            - fidelity(guess, guess + delta)
            + fidelity(guess, guess - delta)
        )

        self.fidelity_eval_count += 4

        # Hessian factor
        h = -0.25 * dF

    # Apply conditioning
    if self.scalar:
        H = scalar_hessian_postprocess(
            self, previous_hessian, h, self.hessian_postprocess_method
        )
        g = (ak / H) * g
    else:
        H = h / np.outer(delta.conj(), delta2)
        H = hessian_postprocess(
            self, previous_hessian, H, self.hessian_postprocess_method
        )
        g = ak * la.solve(H, g, assume_a="her")

    return g, H
