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

print(sol)
```
