##
## tools
##

import numpy as np
import scipy.sparse as sp
from itertools import chain

# general pointwise multiply
def multiply(a, b):
    return a.multiply(b) if sp.issparse(a) else a*b

# make a sparse matrix dense
def ensure_dense(x):
    return x.toarray() if sp.issparse(x) else x

# handles empty data
def vstack(v):
    if len(v) == 0:
        return None
    else:
        return np.vstack(v)

# allows None's and handles empty data
def hstack(v, format='csr'):
    v = [x for x in v if x is not None]
    if len(v) == 0:
        return None
    if any([sp.issparse(x) for x in v]):
        return sp.hstack(v, format=format)
    else:
        return np.hstack(v)

# this assumes row major to align with product
def strides(v):
    if len(v) == 1:
        return np.array([1])
    else:
        return np.r_[1, np.cumprod(v[1:])][::-1]

# concat lists
def chainer(v):
    return list(chain.from_iterable(v))

# split list on boolean condition
def categorize(func, seq):
    true, false = [], []
    for item in seq:
        if func(item):
            true.append(item)
        else:
            false.append(item)
    return true, false

# handles sparse too (dense based on statsmodels version)
def group_sums(x, codes):
    if sp.issparse(x):
        _, K = x.shape
        x = x.tocsc()
        C = max(codes) + 1
        idx = [(x.indptr[i], x.indptr[i+1]) for i in range(K)]
        return np.vstack([
            np.bincount(
                codes[x.indices[i:j]], weights=x.data[i:j], minlength=C
            ) for i, j in idx
        ]).T
    elif x.ndim == 1:
        return np.bincount(codes, weights=x)
    else:
        _, K = x.shape
        return np.vstack([
            np.bincount(codes, weights=x[:, j]) for j in range(K)
        ]).T

# sparsity handled by group_sums
def group_means(x, codes):
    if x.ndim == 1:
        return group_sums(x, codes)/np.bincount(codes)
    else:
        return group_sums(x, codes)/np.bincount(codes)[:, None]
