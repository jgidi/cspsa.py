# cspsa.py
Complex Simultaneous Perturbation Stochastic Approximation

Minimization of real functions depending on many real or complex parameters.
Based upon the original SPSA method, [Spall (1987)](https://ieeexplore.ieee.org/document/4789489), and some derivations presented in [Gidi et. al. (2023)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.108.032409).

# Installation

You can install this package providing the GitHub link to pip. For example,
``` sh
python3 -m pip install git+https://github.com/jgidi/cspsa.py
```

then, you can use this library in python normally,

``` python
import cspsa

def f(x): # Function to minimize
    return  x[0]**2 + (x[1] - 4)**2

guess = [12, 7] # Random value to start iterating from

optimizer = cspsa.SPSA()
sol = optimizer.run(f, guess)

print(sol) # approx np.array([0, 4])
```

# Quickstart

From cspsa you can import the optimizers SPSA and CSPSA. Both take the same arguments.

To define an optimizer you first run

``` python
optimizer = cspsa.SPSA() # or CSPSA()
```

which is equivalent to writing

``` python
import numpy as np

optimizer = cspsa.SPSA(
    # Individual gain parameters (a,b,A,s,t) can be passed; when any are None,
    # the library falls back to the values in the dict `cspsa.DEFAULT_GAINS`.
    a=None,
    b=None,
    A=None,
    s=None,
    t=None,
    # Number of the initial iteration
    init_iter=0,  
    # A function to be called after each iteration, taking (iter, params).
    callback=cspsa.do_nothing,  
    # A function taking `guess` and `update`, and returning the guess for the next iteration
    apply_update=np.add,
    # If a second order update rule should be used.
    second_order=False,
    # If quantum natural preconditioning should be used. Incompatible with `second_order`.
    quantum_natural=False,  
    # If a scalar approximation should be used when computing with `second_order` or `quantum_natural`.
    scalar=False,
    # The hessian postprocessing to use when computing with `second_order` or `quantum_natural`
    hessian_postprocess_method="Gidi",
)
```

Here, `callback` is a function that should take the iteration number and the new parameters found. For instance, you can use it to accumulate the value of the parameters at each iteration as

``` python
params = []
def cb(iter, x):
    params.append(x)

optimizer = cspsa.SPSA(callback=cb)

optimizer.run(f, guess)
# Now `params` contains the evolution of `guess` at each iteration
```
The method `run` also takes a keyword to allow for showing a progress bar,
``` python
optimizer.run(fun, guess, progressbar=True)
```

An alternative approach that allows more control is to use the method `step` instead of `run`, as in

``` python
optimizer = cspsa.SPSA()

params = []
new_guess = guess
num_iter = 100
for _ in range(num_iter):
    new_guess = optimizer.step(f, new_guess)
    params.append(new_guess)
```

## Optimizer properties & advanced usage

### Counters

- `optimizer.iter`: The number of the current iteration. It starts from `optimizer.init_iter` and is incremented after each step.
- `optimizer.iter_count`: How many iterations have been executed.
- `optimizer.function_eval_count`: How many times the objective function (`fun`) has been called.
- `optimizer.fidelity_eval_count`: How many times the fidelity function (used with `quantum_natural`) has been called.

These counters are updated automatically during `step` and `run`.

### Collecting parameters during the run

Use `optimizer.make_params_collector()` to get an empty list that will be populated with the new parameters after each iteration.

Example:

```python
params = optimizer.make_params_collector()
optimizer.run(f, guess, num_iter=100)
# `params` now contains the guesses at each iteration
```

Note: `make_params_collector()` replaces the optimizer's `callback` with a collector wrapping the callback provided when the optimizer was created.

### Restarting an optimizer

Call `optimizer.restart()` to reset the counters and the RNG to the initial state. This sets `iter` back to `init_iter`, zeroes `function_eval_count` and `fidelity_eval_count`, etc

### Second-order and quantum-natural optimization

- To enable second-order preconditioning, create the optimizer with `second_order=True`.
- To enable quantum-natural preconditioning, use `quantum_natural=True`. These two options are mutually exclusive.

When using `quantum_natural`, you must pass a `fidelity` callable to `run` or `step`. The optimizer will call the fidelity function as needed and increment `optimizer.fidelity_eval_count` accordingly.

Example (quantum natural):

```python
def fidelity(x, y):
    # return a scalar fidelity between parameter vectors x and y
    return np.real(np.vdot(x, y))

optimizer = cspsa.CSPSA(quantum_natural=True)
sol = optimizer.run(f, guess, num_iter=200, fidelity=fidelity)
```

### Hessian postprocessing and scalar approximation

When using `second_order` or `quantum_natural`, the optimizer computes a Hessian-like matrix which is postprocessed using the method specified by `hessian_postprocess_method` (default: `"Gidi"`). Supported methods are `"Gidi"` and `"Spall"`.

If `scalar=True` is set, the optimizer uses a scalar approximation for the Hessian instead of a full matrix.

### Callbacks and stopping

The default callback (`do_nothing`) returns nothing. The optimizer treats any non-None return value from the callback as a flag to stop iterating. This means you can implement convergence checks in your callback and stop the optimizer early by returning any value.

Example:

```python
def cb(iter, params):
    # return a non-None value to stop
    if iter > 1000:
        return True

opt = cspsa.SPSA(callback=cb)
opt.run(f, guess, num_iter=10000)
```

### Custom update rule

Instead of requiring a postprocessing function, you can provide a custom `apply_update(guess, update)` when creating the optimizer. This function should return the guess for the next iteration given the current `guess` and the computed `update`.

By default, `apply_update = np.add`, so the next guess is computed as `new_guess = np.add(guess, update)`.

Example (custom update that clips values):

```python
def clip_add(guess, update):
    new = guess + update
    return np.clip(new, -10, 10)

opt = cspsa.SPSA(apply_update=clip_add)
```

### Blocking

The `blocking` feature allows the optimizer to reject updates that do not improve over the last result. The parameter `blocking_tol` allows relaxing the acceptance criteria, which for minimization is
```
f(x_new) - f(x_old) < blocking_tol * np.abs(f(x_old)).
```

It is important to note that blocking increases the number of function evaluations by one per iteration.

Example:

```python
optimizer = cspsa.CSPSA(blocking=True, blocking_tol=1e-3)
sol = optimizer.run(f, guess, num_iter=100)
```
