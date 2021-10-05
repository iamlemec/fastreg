##
## linear regressions
##

import numpy as np
import scipy.sparse as sp

from .tools import (
    hstack, multiply, ensure_dense, group_sums, group_means, chainer,
    block_inverse
)
from .formula import (
    category_indices, design_matrices, parse_tuple, ensure_formula, Categ
)
from .summary import param_table

def ols(
    y=None, x=None, formula=None, data=None, absorb=None, cluster=None,
    hdfe=None, drop='first', extern=None, output='table'
):
    # convert to formula system
    y, x = ensure_formula(x=x, y=y, formula=formula)

    # use hdfe trick
    if hdfe is not None:
        h_trm = parse_tuple(hdfe, convert=Categ)
        x = (x - h_trm) + h_trm # pop to end

    # make design matrices
    y_vec, y_name, x0_mat, x0_names, c_mat, c_names0 = design_matrices(
        y=y, x=x, formula=formula, data=data, drop=drop, extern=extern
    )

    # combine x variables
    x_mat = hstack([x0_mat, c_mat])
    c_names = chainer(c_names0.values())
    x_names = x0_names + c_names

    # data shape
    N = len(data)
    K = len(x_names)

    # use absorption
    if absorb is not None:
        cluster = absorb
        x_mat = ensure_dense(x_mat)
        a_trm = parse_tuple(absorb, convert=Categ)
        a_mat = a_trm.raw(data, extern=extern)
        y_vec, x_mat, keep = absorb_categorical(y_vec, x_mat, a_mat)

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

    # just the betas
    if output == 'beta':
        return beta

    # find residuals
    y_hat = x_mat @ beta
    e_hat = y_vec - y_hat

    # find inv(xpx) somehow
    if hdfe is not None:
        Kh = len(c_names0[h_trm])
        xh_mat, ch_mat = ensure_dense(x_mat[:, :-Kh]), x_mat[:, -Kh:]
        ixr, ixc = block_outer_inverse(xh_mat, ch_mat)
    else:
        ixpx = inv(xpx)

    # find standard errors
    if cluster is not None:
        if absorb is not None:
            c_mat = a_mat[keep, :]
        else:
            c_trm = parse_tuple(cluster, convert=Categ)
            c_mat = c_trm.raw(data, extern=extern)

        # compute sigma
        xe2 = error_sums(x_mat, c_mat, e_hat)
        sigma = ixpx @ xe2 @ ixpx
    else:
        s2 = (e_hat @ e_hat)/(N-K)
        if hdfe is not None:
            sigma = s2*ixr, s2*ixc
        else:
            sigma = s2*ixpx

    # return requested
    if output == 'table':
        return param_table(beta, sigma, y_name, x_names)
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
def error_sums(X, C, e):
    xe = multiply(X, e[:, None])
    codes = category_indices(C)
    xeg = group_sums(xe, codes)
    xe2 = xeg.T @ xeg
    return xe2

##
## absorption
##

def absorb_categorical(y, x, abs):
    N, K = x.shape
    _, A = abs.shape

    # copy so as not to destroy
    y = y.copy()
    x = x.copy()

    # store original means
    avg_y0 = np.mean(y)
    avg_x0 = np.mean(x, axis=0)

    # track whether to drop
    keep = np.ones(N, dtype=np.bool)

    # do this iteratively to reduce data loss
    for j in range(A):
        # create class groups
        codes = category_indices(abs[:, j])

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

    # drop singletons
    y = y[keep]
    x = x[keep, :]

    return y, x, keep
