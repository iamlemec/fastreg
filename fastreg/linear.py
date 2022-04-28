##
## linear regressions
##

import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from functools import reduce
from operator import mul, and_

from .tools import (
    hstack, multiply, ensure_dense, group_sums, group_means, chainer,
    block_inverse, valid_rows
)
from .formula import (
    category_indices, design_matrices, parse_tuple, parse_list, ensure_formula,
    Categ
)
from .summary import param_table

def ols(
    y=None, x=None, formula=None, data=None, absorb=None, cluster=None,
    hdfe=None, stderr=True, drop='first', extern=None, output='table'
):
    # convert to formula system
    y, x = ensure_formula(x=x, y=y, formula=formula)

    # track valid rows
    valid = np.ones(len(data), dtype=bool)

    # use hdfe trick
    if hdfe is not None:
        h_trm = parse_tuple(hdfe, convert=Categ)
        x = (x - h_trm) + h_trm # pop to end

    # don't include absorb
    if absorb is not None:
        a_trm = parse_list(absorb, convert=Categ)
        a_mat = a_trm.raw(data, extern=extern)
        valid &= reduce(and_, (valid_rows(a) for a in a_mat))
        x -= a_trm
        if cluster is None:
            cluster = reduce(mul, a_trm)

    # cluster=False is only to override auto on absorb
    if cluster is False:
        cluster = None

    # fetch cluster var
    if cluster is not None:
        s_trm = parse_tuple(cluster, convert=Categ)
        s_mat = s_trm.raw(data, extern=extern)
        valid &= valid_rows(s_mat)

    # make design matrices
    y_vec, y_name, x0_mat, x0_names, c_mat, c_names0, valid = design_matrices(
        y=y, x=x, formula=formula, data=data, valid0=valid, drop=drop, extern=extern,
        flatten=False, validate=True
    )

    # drop final invalid from cluster and absorb
    if absorb is not None:
        a_mat = [a[valid] for a in a_mat]
    if cluster is not None:
        s_mat = s_mat[valid]

    # combine x variables
    x_mat = hstack([x0_mat, c_mat])
    c_names = chainer(c_names0.values())
    x_names = x0_names + c_names

    # data shape
    N = len(data)
    K = len(x_names)

    # use absorption
    if absorb is not None:
        x_mat = ensure_dense(x_mat)
        y_vec, x_mat, keep = absorb_categorical(y_vec, x_mat, a_mat)
        if cluster is not None:
            s_mat = s_mat[keep, :]

    # linalg tool select
    if sp.issparse(x_mat):
        inv = sp.linalg.inv
        solve = sp.linalg.spsolve
    else:
        inv = np.linalg.inv
        solve = np.linalg.solve

    # find point estimates
    xpx = x_mat.T @ x_mat
    xpy = x_mat.T @ y_vec
    beta = solve(xpx, xpy)

    # just the point estimates
    if output == 'point':
        return beta

    # find residuals
    y_hat = x_mat @ beta
    e_hat = y_vec - y_hat

    # just the point estimate
    if stderr is False:
        if output == 'table':
            return param_table(beta, y_name, x_names)
        elif output == 'dict':
            return {
                'beta': beta,
                'y_name': y_name,
                'x_names': x_names,
                'y_hat': y_hat,
                'e_hat': e_hat,
            }

    # find inv(xpx) somehow
    if hdfe is not None:
        Kh = len(c_names0[h_trm])
        xh_mat, ch_mat = ensure_dense(x_mat[:, :-Kh]), x_mat[:, -Kh:]
        ixr, ixc = block_outer_inverse(xh_mat, ch_mat)
    else:
        ixpx = ensure_dense(inv(xpx))

    # compute classical sigma hat
    if stderr is True or hdfe is not None:
        s2 = (e_hat @ e_hat)/(N-K)

    # compute Xe moment
    if cluster is not None or type(stderr) is str:
        xe_mat = multiply(x_mat, e_hat[:, None])

    # find standard errors
    if cluster is not None:
        xe2 = error_sums(xe_mat, s_mat)
        sigma = ixpx @ xe2 @ ixpx
    elif hdfe is not None:
        sigma = s2*ixr, s2*ixc
    elif stderr is True:
        sigma = s2*ixpx
    elif type(stderr) is str:
        hc, = map(int, re.match(r'hc([0-3])', stderr).groups())
        sigma = hcn_stderr(hc, x_mat, xe_mat, ixpx)

    # return requested
    if output == 'table':
        return param_table(beta, y_name, x_names, sigma=sigma)
    elif output == 'dict':
        return {
            'beta': beta,
            'sigma': sigma,
            'y_name': y_name,
            'x_names': x_names,
            'y_hat': y_hat,
            'e_hat': e_hat,
        }

##
## standard errors
##

# inv(X.T @ X) when X = [X D] and D is sparse and diagonal
def block_outer_inverse(X, D):
    A = X.T @ X
    B = X.T @ D
    C = D.T @ X
    d = D.power(2).sum(axis=0).getA1()
    return block_inverse(A, B, C, d)

# from cameron and miller
def error_sums(xe, c):
    codes, _ = category_indices(c, dropna=True)
    xeg = group_sums(xe, codes)
    xe2 = xeg.T @ xeg
    return xe2

# handles hc in (0, 1, 2, 3)
def hcn_stderr(hc, x, xe, ixpx):
    N, K = x.shape
    if hc < 2:
        xeh = xe
    else:
        xq = x @ np.linalg.cholesky(ixpx)
        hii = np.sum(xq**2, axis=1)
        hinv = 1/(1-hii)
        if hc == 2:
            hinv = np.sqrt(hinv)
        xeh = multiply(xe, hinv[:, None])
    sigma = ixpx @ (xeh.T @ xeh) @ ixpx
    if hc == 1:
        sigma *= N/(N-K)
    return sigma

##
## absorption
##

# will absorb null (-1) values together
def absorb_categorical(y, x, abs):
    N, K = x.shape

    # copy so as not to destroy
    y = y.copy()
    x = x.copy()

    # store original means
    avg_y0 = np.mean(y)
    avg_x0 = np.mean(x, axis=0)

    # track whether to drop
    keep = np.ones(N, dtype=bool)

    # do this iteratively to reduce data loss
    for a in abs:
        # create class groups
        codes, _ = category_indices(a, dropna=True)

        # perform differencing on y
        avg_y = group_means(y, codes)
        y -= avg_y[codes]

        # perform differencing on x
        avg_x = group_means(x, codes)
        x -= avg_x[codes, :]

        # detect singletons
        multi = np.bincount(codes) > 1
        keep &= multi[codes]

    # recenter means
    y += avg_y0
    x += avg_x0[None, :]

    # drop invalid
    y = y[keep]
    x = x[keep, :]

    return y, x, keep
