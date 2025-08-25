#!/usr/bin/env python3

import numpy as np
from copy import copy
from tqdm import tqdm
from scipy import linalg as la
from typing import Callable, Sequence

from .defaults import (
    DEFAULT_NUM_ITER,
    DEFAULT_GAINS,
    DEFAULT_COMPLEX_PERTURBATIONS,
    DEFAULT_HESSIAN_POSTPROCESS_METHOD,
    DEFAULT_HESSIAN_POSTPROCESS_TOL,
    do_nothing,
)


class CSPSA:
    def __init__(
        self,
        a: float | None = None,
        b: float | None = None,
        A: float | None = None,
        s: float | None = None,
        t: float | None = None,
        init_iter: int = 0,
        callback: Callable = do_nothing,
        apply_update: Callable = np.add,
        perturbations: Sequence = DEFAULT_COMPLEX_PERTURBATIONS,
        maximize: bool = False,
        scalar: bool = False,
        second_order: bool = False,
        quantum_natural: bool = False,
        hessian_postprocess_method: str = DEFAULT_HESSIAN_POSTPROCESS_METHOD,
        seed: int | None = None,
    ):
        self.a_precond = 1.0 if a is None else a
        self.a = DEFAULT_GAINS["a"] if a is None else a
        self.b = DEFAULT_GAINS["b"] if b is None else b
        self.A = DEFAULT_GAINS["A"] if A is None else A
        self.s = DEFAULT_GAINS["s"] if s is None else s
        self.t = DEFAULT_GAINS["t"] if t is None else t
        self.init_iter = init_iter
        self.sign = 2 * maximize - 1
        self.apply_update = apply_update
        self.perturbations = perturbations
        self.outer_callback = callback
        self.seed = seed
        self.rng = np.random.default_rng(seed)

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
        assert not (self.scalar and not self.is_preconditioned), errmsg

    # Minimal callback action: Check for returned value and set self.stop
    def _callback(self, *args):
        self.stop = self.outer_callback(*args) is not None

    def callback(self, iter, guess):
        self._callback(iter, guess)

    def make_params_collector(self):
        params = []

        def wrapper(iter, guess):
            params.append(guess)
            self._callback(iter, guess)

        self.callback: Callable = wrapper

        return params

    @property
    def is_preconditioned(self) -> bool:
        return self.second_order or self.quantum_natural

    @property
    def iter_count(self):
        return self.iter - self.init_iter

    def restart(self):
        self.stop = False
        self.iter = self.init_iter
        self.function_eval_count = 0
        self.fidelity_eval_count = 0
        self.rng = np.random.default_rng(self.seed)

    def _stepsize_and_pert(self):
        a = self.a
        A = self.A
        b = self.b
        s = self.s
        t = self.t

        if self.is_preconditioned:
            a = self.a_precond

        ak = a / (self.iter + 1 + A) ** s
        bk = b / (self.iter + 1) ** t

        return ak, bk

    def default_hessian(self, guess) -> np.ndarray:
        if self.scalar:
            return np.array([[1.0]])
        else:
            return np.eye(len(guess))

    def step(
        self,
        fun: Callable,
        guess: np.ndarray,
        previous_hessian: np.ndarray | None = None,
        fidelity: Callable | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not self.is_preconditioned:
            return self._first_order_step(fun, guess)
        else:
            return self._preconditioned_step(fun, guess, previous_hessian, fidelity)

    def run(
        self,
        fun: Callable,
        guess: np.ndarray,
        num_iter: int = DEFAULT_NUM_ITER,
        progressbar: bool = False,
        initial_hessian=None,
        fidelity=None,
    ) -> np.ndarray:
        new_guess = np.copy(guess)
        iterator = range(self.init_iter, self.init_iter + num_iter)
        iterator = tqdm(iterator, disable=not progressbar)

        if not self.is_preconditioned:
            for _ in iterator:
                new_guess = self.step(fun, new_guess)
                if self.stop:
                    break

        # Preconditioning
        else:
            H = copy(initial_hessian)
            for _ in iterator:
                new_guess, H = self.step(fun, new_guess, H, fidelity)
                if self.stop:
                    break

        return new_guess

    # =============== First order
    def _first_order_step(self, fun: Callable, guess: np.ndarray) -> np.ndarray:
        ak, bk = self._stepsize_and_pert()

        delta = bk * self.rng.choice(self.perturbations, len(guess))
        df = fun(guess + delta) - fun(guess - delta)
        self.function_eval_count += 2

        update = self.sign * 0.5 * ak * df / delta.conj()
        new_guess = self.apply_update(guess, update)

        self.callback(self.iter, new_guess)
        self.iter += 1

        return new_guess

    # =============== Preconditioning
    def _preconditioned_step(
        self,
        fun: Callable,
        guess: np.ndarray,
        previous_hessian: np.ndarray | None = None,
        fidelity: Callable | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if previous_hessian is None:
            previous_hessian = self.default_hessian(guess)

        update, hessian = self._preconditioned_update(
            fun, guess, previous_hessian, fidelity
        )

        new_guess = self.apply_update(guess, update)

        self.callback(self.iter, new_guess)
        self.iter += 1

        return new_guess, hessian

    def _preconditioned_update(
        self,
        fun: Callable,
        guess: np.ndarray,
        previous_hessian: np.ndarray,
        fidelity: Callable | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ak, bk = self._stepsize_and_pert()

        delta = bk * self.rng.choice(self.perturbations, len(guess))
        delta2 = bk * self.rng.choice(self.perturbations, len(guess))

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
            H = np.array([[h]])
        else:
            H = h / np.outer(delta.conj(), delta2)

        H = self._hessian_postprocess(
            previous_hessian, H, self.hessian_postprocess_method
        )
        g = self.sign * ak * la.solve(H, g, assume_a="her")

        return g, H

    def _hessian_postprocess(
        self,
        Hprev: np.ndarray,
        H: np.ndarray,
        tol: float = DEFAULT_HESSIAN_POSTPROCESS_TOL,
    ) -> np.ndarray:
        k = self.iter
        I = np.eye(H.shape[0])

        H = (H + H.T.conj()) / 2
        if self.hessian_postprocess_method == "Gidi":
            H = la.sqrtm(H @ H.T.conj() + tol * I)
            H = (k * Hprev + H) / (k + 1)
        elif self.hessian_postprocess_method == "Spall":
            H = (k * Hprev + H) / (k + 1)
            H = la.sqrtm(H @ H.T.conj()) + tol * I
        else:
            raise ValueError("Hessian postprocess method should be 'Gidi' or 'Spall'.")
            

        return H
