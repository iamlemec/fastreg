##
## tools
##

import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
from itertools import chain
from functools import partial

# general pointwise multiply
def multiply(a, b):
    return a.multiply(b) if sp.issparse(a) else a*b

# make a sparse matrix dense
def ensure_dense(x):
    return x.toarray() if sp.issparse(x) else x

# return vector or matrix diagonal
def maybe_diag(x):
    if x.ndim == 1:
        return x
    else:
        return x.diagonal()

# handles empty data
def vstack(v, format='csr'):
    v = [x for x in v if x is not None]
    if len(v) == 0:
        return None
    elif any([sp.issparse(x) for x in v]):
        return sp.vstack(v, format=format)
    else:
        return np.vstack(v)

# allows None's and handles empty data
def hstack(v, format='csr'):
    v = [x for x in v if x is not None]
    if len(v) == 0:
        return None
    elif any([sp.issparse(x) for x in v]):
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

# block inversion with diagonal d
def block_inverse(A, B, C, d, inv=la.inv):
    d1 = 1/d
    d1l, d1r = d1[:, None], d1[None, :]
    A1 = inv(A - (B*d1r) @ C)
    Ai = A1
    di = d1 + np.sum((d1l*C)*(A1 @ (B*d1r)).T, axis=1)
    return Ai, di

##
## function tools
##

# decorator with optional flags
def decorator(decor0):
    def decor1(func=None, *args, **kwargs):
        if func is None:
            def decor2(func1):
                return decor0(func1, *args, **kwargs)
            return decor2
        else:
            return decor0(func, *args, **kwargs)
    return decor1

def func_name(func, anon='f'):
    fname = func.__name__
    if anon is not None and fname == '<lambda>':
        fname = anon
    return fname

def func_args(name, *args, **kwargs):
    astr = ','.join([f'{a}' for a in args])
    kstr = ','.join([f'{k}={v}' for k, v in kwargs.items()])
    sig = '|'.join(filter(len, [astr, kstr]))
    return f'{name}({sig})'

def func_disp(func, name=None):
    name = func_name(func) if name is None else name
    return partial(func_args, name)
