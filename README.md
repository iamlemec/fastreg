# fastreg

fast sparse regressions. good for high-dimensional fixed effects

### Usage

Regress `y` on `x` given `pandas` DataFrame `data`:
```
linear.ols('y', ['x'], data)
```

Regress `y` on `x1`, `x2`, categorical `id1`, and all combinations of categoricals `id2` and `id3`:
```
linear.ols('y', ['x1', 'x2'], ['id1', ('id2', 'id3')], data)
```
