# fastreg

Fast sparse regressions. Good for high-dimensional fixed effects.

**Also**: experimental work on doing maximum likelihood estimation with JAX.

### Usage

Regress `y` on `x1` and `x2` given `pandas` DataFrame `data`:
```
linear.ols(y='y', x=['x1', 'x2'], data=data)
```

Regress `y` on `x1`, `x2`, categorical `id1`, and categorical `id2`:
```
linear.ols(y='y', x=['x1', 'x2'], fe=['id1', 'id2'], data=data)
```

Regress `y` on `x1`, `x2`, categorical `id1`, and all combinations of categoricals `id2` and `id3`:
```
linear.ols(y='y', x=['x1', 'x2'], fe=['id1', ('id2', 'id3')], data=data)
```

### Experimental

There's a maximum likelihood estimation routine in `general.py` called `maxlike`. Just give this a function that computes the mean log likelihood and it'll take care of the rest. This is then specialized into a generalized linear model routine called `glm`, which accepts link and loss functions along with data. I've provided implementations for `logit`, `poisson`, `zero_inflated_poisson`, `negative_binomial`, and `ordinary_least_squares`. These all use the same syntax as `linear.ols`.
