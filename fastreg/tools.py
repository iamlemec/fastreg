##
## tools
##

import numpy as np
import pandas as pd
import numpy.linalg as la
import scipy.sparse as sp
from itertools import chain, accumulate
from functools import partial

# general pointwise multiply
def multiply(a, b):
    return a.multiply(b) if sp.issparse(a) else a*b

# make a sparse matrix dense
def ensure_dense(x):
    return x.toarray() if sp.issparse(x) else x

# make a row vector into a matrix, maybe
def atleast_2d(x, axis=0):
    if x.ndim < 2:
        if axis == 0:
            return x.reshape((-1, 1))
        else:
            return x.reshape((1, -1))
    else:
        return x

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

# do isnan on general ndarrays
def fillna(x, v=0):
    return np.where(np.isnan(x), v, x)

# try to handle dense and various sparse formats
def valid_rows(x):
    fmt = x.format if sp.issparse(x) else None
    if fmt is None:
        null = ~pd.isnull(x)
        return null if x.ndim == 1 else null.any(axis=1)
    elif fmt == 'csr':
        N, _ = x.shape
        nidx, = np.nonzero(np.isnan(x.data))
        rows = np.unique(np.digitize(nidx, x.indptr)-1)
        null = np.isin(np.arange(N), rows)
    else:
        print(f'valid_rows: unsupported format "{fmt}"')
    return ~null

def all_valid(*mats):
    return np.vstack([m for m in mats if m is not None]).all(axis=0)

# list based cumsum
def cumsum(x):
    return list(accumulate(x))

# split by sizes rather than boundaries
def split_size(x, s):
    b = cumsum(s)
    return np.split(x, b)

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

##
## semi-defunct tools
##

def category_indices_numeric(vals, return_labels=False):
    # get unique indices
    labs, indx = np.unique(vals, axis=0, return_inverse=True)

    # patch in nans as -1
    nan_loc = np.flatnonzero(np.isnan(labs).any(axis=-1))
    if len(nan_loc) > 0:
        labs1 = np.delete(labs, nan_loc, axis=0)
        indx1 = np.where(np.isin(indx, nan_loc), -1, indx)
        _, indx2 = np.unique(indx1, return_inverse=True)
        indx, labs = indx2-1, labs1

    if return_labels:
        return indx, list(map(tuple, labs))
    else:
        return indx
