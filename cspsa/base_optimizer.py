#!/usr/bin/env python3

import numpy as np
from copy import copy
from tqdm import tqdm
from scipy import linalg as la
from typing import Callable, Sequence

from .defaults import *

def do_nothing(*args):
    pass

def identity(x):
    return x

class CSPSA:
    def __init__(
        self,
        gains: dict = DEFAULT_GAINS,
        init_iter: int = 0,
        callback: Callable = do_nothing,
        postprocessing: Callable = identity,
        perturbations: Sequence = DEFAULT_COMPLEX_PERTURBATIONS,
        scalar: bool = False,
        second_order: bool = False,
        quantum_natural: bool = False,
        hessian_postprocess_method: str = DEFAULT_HESSIAN_POSTPROCESS_METHOD,
    ):

        self.gains = copy(gains)
        self.init_iter = init_iter
        self.perturbations = perturbations
        self.postprocessing = postprocessing
        self.outer_callback = callback

        # Preconditioning
        self.scalar = scalar
        self.second_order = second_order
        self.quantum_natural = quantum_natural
        self.hessian_postprocess_method = hessian_postprocess_method

        self.restart()
        self._check_args()

    # TODO
    def _check_args(self):
        errmsg = "Can't set both 'second_order=True' and 'quantum_natural=True'"
        assert not (self.second_order and self.quantum_natural), errmsg

        errmsg = "Can't set 'scalar=True' if not using second_order or quantum_natural"
        preconditioned = self.second_order or self.quantum_natural
        assert not (self.scalar and (not preconditioned)), errmsg

    def callback(self, guess):
        self.stop = self.outer_callback(guess) is not None

    def make_collector(self):
        params = []

        def wrapper(self, guess):
            params.append(guess)
            self.stop = self.outer_callback(guess) is not None

        self.callback = wrapper

        return params

    @property
    def iter_count(self):
        return self.iter - self.init_iter

    def restart(self):
        self.stop = False
        self.iter = self.init_iter
        self.function_eval_count = 0
        self.fidelity_eval_count = 0

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

    def default_hessian(self, guess) -> np.ndarray:
        return np.eye(len(guess))

    def step(
        self,
        fun: Callable,
        guess: np.ndarray,
        previous_hessian: np.ndarray | None = None,
        fidelity: Callable | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:

        first_order = not (self.second_order or self.quantum_natural)
        if first_order:
            return first_order_step(self, fun, guess)
        else:
            return preconditioned_step(self, fun, guess, previous_hessian, fidelity)

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
        iterator = range(self.init_iter, self.init_iter + num_iter)
        iterator = tqdm(iterator, disable=not progressbar)

        first_order = not (self.second_order or self.quantum_natural)
        if first_order:
            for _ in iterator:
                new_guess = first_order_step(self, fun, new_guess)
                if self.stop: break

        # Preconditioning
        else:
            H = self.default_hessian(guess, initial_hessian)
            for _ in iterator:
                new_guess, H = preconditioned_step(self, fun, new_guess, H, fidelity)
                if self.stop: break

        return new_guess

def first_order_step(self: "CSPSA",
                     fun: Callable,
                     guess: np.ndarray):

    ak, bk = self._stepsize_and_pert()

    delta = bk * np.random.choice(self.perturbations, len(guess))
    df = fun(guess + delta) - fun(guess - delta)

    self.function_eval_count += 2

    g = 0.5 * ak * df / delta.conj()

    return g

# =============== Preconditioning

def preconditioned_step(
        self,
        fun: Callable,
        guess: np.ndarray,
        previous_hessian: np.ndarray | None = None,
        fidelity: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray]:

    if previous_hessian is None:
        previous_hessian = self.default_hessian(guess)

    update, hessian = preconditioned_update(
        self, fun, guess, previous_hessian, fidelity
    )

    new_guess = guess - update
    new_guess = self.postprocessing(new_guess)

    self.callback(self, new_guess)
    self.iter += 1

    return new_guess, hessian

def preconditioned_update(
    self: "CSPSA",
    fun: Callable,
    guess: np.ndarray,
    previous_hessian: np.ndarray,
    fidelity: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray]:

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
        H = np.array([[ h ]])
    else:
        H = h / np.outer(delta.conj(), delta2)

    H = hessian_postprocess(
        self, previous_hessian, H, self.hessian_postprocess_method
    )
    g = ak * la.solve(H, g, assume_a="her")

    return g, H

def hessian_postprocess(
    self: "CSPSA",
    Hprev: np.ndarray,
    H: np.ndarray,
    method: str = DEFAULT_HESSIAN_POSTPROCESS_METHOD,
    tol: float = DEFAULT_HESSIAN_POSTPROCESS_TOL,
) -> np.ndarray:

    k = self.iter
    I = np.eye(H.shape[0])

    H = (H + H.T.conj()) / 2
    if method == "Gidi":
        H = la.sqrtm(H @ H.T.conj() + tol * I)
        H = (k * Hprev + H) / (k + 1)
    elif method == "Spall":
        H = (k * Hprev + H) / (k + 1)
        H = la.sqrtm(H @ H.T.conj()) + tol * I
    else:
        msg = f"Hessian postproces method should 'Gidi' or 'Spall'. Got {method}."
        raise Exception(msg)

    return H
