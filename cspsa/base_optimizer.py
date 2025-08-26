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

        # Preconditioning
        self.scalar = scalar
        self.second_order = second_order
        self.quantum_natural = quantum_natural
        self.hessian_postprocess_method = hessian_postprocess_method
        self.H = None

        self.restart()
        self._check_args()

    # Private methods
    def _check_args(self):
        errmsg = "Can't set both 'second_order=True' and 'quantum_natural=True'"
        assert not (self.second_order and self.quantum_natural), errmsg

        errmsg = "Can't set 'scalar=True' if not using second_order or quantum_natural"
        assert not (self.scalar and not self.preconditioned), errmsg

    def _sample_delta(self, bk: float, size: int) -> np.ndarray:
        return bk * self.rng.choice(self.perturbations, size)

    def _stepsize_and_pert(self):
        a = self.a
        A = self.A
        b = self.b
        s = self.s
        t = self.t

        if self.preconditioned:
            a = self.a_precond

        ak = a / (self.iter + 1 + A) ** s
        bk = b / (self.iter + 1) ** t

        return ak, bk

    def _default_hessian(self, guess) -> np.ndarray:
        if self.scalar:
            return np.array([[1.0]])
        else:
            return np.eye(len(guess))

    def _first_order_step(self, fun: Callable, guess: np.ndarray) -> np.ndarray:
        ak, bk = self._stepsize_and_pert()

        delta = self._sample_delta(bk, len(guess))
        df = self._compute_difference(fun, guess, delta)
        update = self.sign * ak * df / delta.conj()
        new_guess = self.apply_update(guess, update)

        self.iter += 1
        self.callback(self.iter, new_guess)

        return new_guess

    def _preconditioned_step(
        self,
        fun: Callable,
        guess: np.ndarray,
        fidelity: Callable | None = None,
    ) -> np.ndarray:
        if self.H is None:
            self.H = self._default_hessian(guess)

        update, hessian = self._preconditioned_update(fun, guess, fidelity)
        self.H = hessian

        new_guess = self.apply_update(guess, update)

        self.iter += 1
        self.callback(self.iter, new_guess)

        return new_guess

    def _compute_difference(
        self, fun: Callable, guess: np.ndarray, delta: np.ndarray
    ) -> float:
        df = 0.5 * (fun(guess + delta) - fun(guess - delta))
        self.function_eval_count += 2
        return df

    def _preconditioned_update(
        self,
        fun: Callable,
        guess: np.ndarray,
        fidelity: Callable | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ak, bk = self._stepsize_and_pert()

        delta = self._sample_delta(bk, len(guess))
        delta2 = self._sample_delta(bk, len(guess))

        df = self._compute_difference(fun, guess, delta)
        g = df / delta.conj()

        if self.second_order:
            dfp = self._compute_difference(fun, guess + delta2, delta)
            h = dfp - df
        else:
            errmsg = "For Quantum Natural optimization, you must provide the fidelity"
            assert fidelity is not None, errmsg

            dF = (
                fidelity(guess, guess + delta + delta2)
                - fidelity(guess, guess - delta + delta2)
                - fidelity(guess, guess + delta)
                + fidelity(guess, guess - delta)
            )
            self.fidelity_eval_count += 4
            h = -0.25 * dF

        # Apply conditioning
        if self.scalar:
            H = np.array([[h / bk**2]])
        else:
            H = h / np.outer(delta.conj(), delta2)

        H = self._hessian_postprocess(self.H, H)
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
            msg = f"Hessian postprocess method should be 'Gidi' or 'Spall'. Got {self.hessian_postprocess_method}."
            raise ValueError(msg)

        return H

    # Public methods
    def restart(self):
        self.stop = False
        self.iter = self.init_iter
        self.function_eval_count = 0
        self.fidelity_eval_count = 0
        self.rng = np.random.default_rng(self.seed)
        self.H = None

    def callback(self, iter, guess):
        # Invoke the user-defined callback and set the stop flag if needed
        self.stop = self.outer_callback(iter, guess) is not None

    def make_params_collector(self):
        # Make a callback that collects parameters at each iteration
        # and wraps the user-defined callback
        params = []

        def collector(iter, guess):
            params.append(guess)
            self.stop = self.outer_callback(iter, guess) is not None

        self.callback = collector
        return params

    @property
    def preconditioned(self) -> bool:
        return self.second_order or self.quantum_natural

    @property
    def iter_count(self):
        return self.iter - self.init_iter

    def step(
        self,
        fun: Callable,
        guess: np.ndarray,
        fidelity: Callable | None = None,
    ) -> np.ndarray:
        if not self.preconditioned:
            return self._first_order_step(fun, guess)
        else:
            return self._preconditioned_step(fun, guess, fidelity)

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

        if not self.preconditioned:
            for _ in iterator:
                new_guess = self.step(fun, new_guess)
                if self.stop:
                    break

        # Preconditioning
        else:
            if initial_hessian is not None:
                self.H = copy(initial_hessian)

            for _ in iterator:
                new_guess = self.step(fun, new_guess, fidelity=fidelity)
                if self.stop:
                    break

        return new_guess

    def calibrate_b(self, fun, guess, num_samples=10):
        meas = [fun(guess) for _ in range(num_samples)]
        self.function_eval_count += num_samples

        b = max(float(np.std(meas)), 1e-5)

        self.b = b
        return b

    def calibrate_a(
        self,
        fun: Callable,
        guess: np.ndarray,
        num_samples: int = 20,
        target_stepsize: float = 1,
    ) -> float:
        # Use active perturbation size
        _, bk = self._stepsize_and_pert()

        # Compute gradient magnitudes
        mags = []
        for _ in range(num_samples // 2):
            delta = self._sample_delta(bk, guess.size)
            df = self._compute_difference(fun, guess, delta)
            mags.append(abs(df / bk))

        # Compute absolute size ak for current iteration
        # and translate for the value of raw a
        ak = target_stepsize / np.median(mags)
        a = ak * (self.A + self.iter + 1) ** self.s
        a = float(a)

        # Set the new value
        self.a = a
        self.a_precond = a

        return a
