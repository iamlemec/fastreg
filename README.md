# fastreg

Fast sparse regressions. Good for high-dimensional fixed effects. Supports OLS, GLM, and poisson.

### Usage

See `benchmark.ipynb` for complete examples.

Regress `y` on `x1` and `x2` given `pandas` DataFrame `data`:
```
linear.ols('y', ['x1', 'x2'], data)
```

Regress `y` on `x1`, `x2`, categorical `id1`, and categorical `id2`:
```
linear.ols('y', ['x1', 'x2'], ['id1', 'id2'], data)
```

Regress `y` on `x1`, `x2`, categorical `id1`, and all combinations of categoricals `id2` and `id3`:
```
linear.ols('y', ['x1', 'x2'], ['id1', ('id2', 'id3')], data)
```

For poisson regressions, the syntax is the same, but use `general.poisson`. Generalized linear models provided with `general.glm`, but must provide analytic derivatives of link and loss functions for standard errors.
