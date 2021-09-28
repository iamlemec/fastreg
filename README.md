# fastreg

Fast sparse regressions. Good for high-dimensional fixed effects.

**Also**: experimental work on doing maximum likelihood estimation and generalized linear models with JAX (see `general.py`).

### Install

To install directly from GitHub, just run:
``` bash
pip install git+https://github.com/iamlemec/fastreg
```
Alternatively, you can clone this repository locally and run
``` bash
pip install -e .
```

Optionally, for the maximum likelihood routines, you'll need `jax` (and `jaxlib`) as well. See [here](https://github.com/google/jax) for detailed instructions.

### Usage

First import the necessary functions
``` python
import fastreg.linear as lin
import fastreg.testing as test
from fastreg.formula import I, R, C
```

Create some testing data
``` python
data = test.dataset(N=100_000, K1=10, K2=100, models='linear')
```

Regress `y` on `1`, `x1`, and `x2` given `pandas` DataFrame `data`:
``` python
fr.ols(y='y', x=I+R('x1')+R('x2'), data=data)
```

Regress `y` on `1`, `x1`, `x2`, categorical `id1`, and categorical `id2`:
``` python
fr.ols(y='y', x=I+R('x1')+R('x2')+C('id1')+C('id2'), data=data)
```

Regress `y` on `1`, `x1`, `x2`, and all combinations of categoricals `id1` and `id2`:
``` python
fr.ols(y='y', x=I+R('x1')+R('x2')+C('id1')*C('id2'), data=data)
```
Note that `*` is analogous to `:` in R-style syntax.

Instead of passing `y` and `x`, you can also pass an R-style formula string to `formula`, as in:
``` python
fr.ols(formula='y ~ 1 + x1 + x2 + C(id1):C(id2)', data=data)
```

There's even a third intermediate option using lists and tuples:
``` python
fr.ols(y='y', x=['1', 'x1', 'x2', (C('id1'), C('id2'))], data=data)
```

Right now, categorical coding schemes other than treatment are not supported. You can pass a list of column names to `cluster` to cluster standard errors on those variables. You can also pass a list of columns to `absorb` to absorb those variables a la Stata's `areg`. This will automatically cluster on those variables as well.

### Experimental

There's a maximum likelihood estimation routine in `general.py` called `maxlike`. Just give this a function that computes the mean log likelihood and it'll take care of the rest. This is then specialized into a generalized linear model routine called `glm`, which accepts link and loss functions along with data. I've provided implementations for `logit`, `poisson`, `negbin`, `zinf_poisson`, `zinf_negbin`, and `ols`. These all use the same syntax as `linear.ols`.
