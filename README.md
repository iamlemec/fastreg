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

Regress `y` on `x1` and `x2` given `pandas` DataFrame `data`:
``` python
linear.ols(y='y', x=['x1', 'x2'], data=data)
```

Regress `y` on `x1`, `x2`, categorical `id1`, and categorical `id2`:
``` python
linear.ols(y='y', x=['x1', 'x2'], fe=['id1', 'id2'], data=data)
```

Regress `y` on `x1`, `x2`, categorical `id1`, and all combinations of categoricals `id2` and `id3`:
``` python
linear.ols(y='y', x=['x1', 'x2'], fe=['id1', ('id2', 'id3')], data=data)
```

Instead of passing `y`, `x`, and `fe`, you can also pass an R-style formula string to `formula`, as in:
``` python
linear.ols(formula='y ~ x1 + x2 + C(id1) + C(id2)', data=data)
```
Right now, coding schemes other than treatment and mixing continuous and categorical variables in one term are not supported.

You can pass a list of column names to `cluster` to cluster standard errors on those variables. You can also pass a list of columns to `absorb` to absorb those variables a la Stata's `areg`. This will automatically cluster on those variables as well.

### Experimental

There's a maximum likelihood estimation routine in `general.py` called `maxlike`. Just give this a function that computes the mean log likelihood and it'll take care of the rest. This is then specialized into a generalized linear model routine called `glm`, which accepts link and loss functions along with data. I've provided implementations for `logit`, `poisson`, `zero_inflated_poisson`, `negative_binomial`, and `ordinary_least_squares`. These all use the same syntax as `linear.ols`.
