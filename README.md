# cspsa.py
Complex Simultaneous Perturbation Stochastic Approximation

# Installation

You can install this package providing the Github link to pip. For example,
``` sh
python3 -m pip install git+https://github.com/jgidi/cspsa.py
```

then, you can use this library in python normally,

``` python
import cspsa

optimizer = cspsa.SPSA()

sol = optimizer.run(fun=lambda x: x[0]**2 + (x[1] - 4)**2, guess=[4, 8])

print(sol) # approx [0, 4]
```

# Examples

From cspsa you can import the optimizers SPSA and CSPSA. Both take the same arguments.

To define an optimizer you first run

``` python
optimizer = cspsa.SPSA() # or CSPSA()
```

which is equivalent to writing

``` python
optimizer = cspsa.SPSA(num_iter = cspsa.DEFAULT_NUM_ITER, # number of iterations
                       gains = cspsa.DEFAULT_GAINS, # Dictionary with the set of gain parameters
                       init_iter = 0, # Number of the initial iteration
                       callback = lambda i,x : None, # A function to be called after each step.
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
