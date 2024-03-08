# cspsa.py
Complex Simultaneous Perturbation Stochastic Approximation

Minimization of real functions depending on many real or complex parameters.
Based upon the original SPSA method, [Spall (1987)](https://ieeexplore.ieee.org/document/4789489), and some derivations presented in [Gidi et. al. (2023)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.108.032409).


**NOTE**: Currently only SPSA and CSPSA are implemented. Quantum-natural methods will come soon.

# Installation

You can install this package providing the Github link to pip. For example,
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

# Examples

From cspsa you can import the optimizers SPSA and CSPSA. Both take the same arguments.

To define an optimizer you first run

``` python
optimizer = cspsa.SPSA() # or CSPSA()
```

which is equivalent to writing

``` python
optimizer = cspsa.SPSA(gains = cspsa.DEFAULT_GAINS, # Dictionary with the set of gain parameters
                       init_iter = 0, # Number of the initial iteration
                       callback = cspsa.do_nothing, # A function to be called after each iteration, taking (iter, params).
                       apply_update = cspsa.np.add, # A function taking `guess` and `update`, and returning the guess for the next iteration
                       second_order = False,  # If a second order update rule should be used.
                       quantum_natural = False, # If quantum natural preconditioning should be used. Incompatible with `second_order`.
                       scalar: bool = False,   # If a scalar approximation should be used when computing with `second_order` or `quantum_natural`.
                       hessian_postprocess_method = "Gidi", # The hessian postprocessing to use when computing with `second_order` or `quantum_natural`
                       )
```

Here, `gains` is a dictionary which controls the convergence of the optimization methods. If a dictionary is provided where some of the gain parameters are not specified, they will be taken from `cspsa.DEFAULT_GAINS`. The asymptotic set of gain parameters is also available as `cspsa.ASYMPTOTIC_GAINS`.

Also, `callback` is a function that you can specify, taking the iteration number and the new parameters found, to perform any task of your liking. For example, you can use it to accumulate the value of the parameters at each iteration as

``` python
params = []
def cb(iter, x):
    params.append(x)

optimizer = cspsa.SPSA(callback=cb)

optimizer.run(fun, guess) # Assuming you have `fun` and `guess` defined
```

An alternative approach would be to use the method `step` intead of `run`, as in

``` python
optimizer = cspsa.SPSA()

params = []
new_guess = guess
for iter in range(optimizer.num_iter):
    new_guess = optimizer.step(fun, new_guess)
    params.append(new_guess)
```

However, if you use the default method `run`, you have niceties such as showing a progress bar for log-running tasks

``` python
optimizer.run(fun, guess, progressbar=True)
```

Finally, note that you can restart the iteration count of an optimizer by running

``` python
optimizer.restart()
```
