##
## linear regressions
##

import numpy as np
import scipy.sparse as sp

from .tools import hstack, multiply, ensure_dense, group_sums, group_means
from .formula import category_indices, design_matrices, parse_tuple
from .summary import param_table

def ols(
    y=None, x=None, formula=None, data=None, absorb=None, cluster=None,
    drop='first', method='solve', output='table'
):
    # make design matrices
    y_vec, y_name, x0_mat, x0_names, c_mat, c_names = design_matrices(
        y=y, x=x, formula=formula, data=data, drop=drop
    )

    # combine x variables
    x_mat = hstack([x0_mat, c_mat])
    x_names = x0_names + c_names

    # data shape
    N = len(y_vec)
    K = len(x_names)

    # use absorption
    if absorb is not None:
        cluster = absorb
        x_mat = ensure_dense(x_mat)
        a_mat = parse_tuple(absorb).raw(data)
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
    ixpx = inv(xpx)
    if method == 'inv':
        beta = ixpx @ xpy
    elif method == 'solve':
        beta = solve(xpx, xpy)

    # just the betas
    if output == 'beta':
        return beta

    # find residuals
    y_hat = x_mat @ beta
    e_hat = y_vec - y_hat

    # find standard errors
    if cluster is not None:
        if absorb is not None:
            c_mat = a_mat[keep, :]
        else:
            c_mat = parse_tuple(cluster).raw(data)

        # from cameron and miller
        xe = multiply(x_mat, e_hat[:, None])
        codes = category_indices(c_mat)
        xeg = group_sums(xe, codes)
        xe2 = xeg.T @ xeg
        sigma = ixpx @ xe2 @ ixpx
    else:
        s2 = (e_hat @ e_hat)/(N-K)
        sigma = s2*ixpx

    if output == 'table':
        return param_table(beta, sigma, y_name, x_names)
    else:
        return {
            'beta': beta,
            'sigma': sigma,
            'y_hat': y_hat,
            'e_hat': e_hat,
            'y_name': y_name,
            'x_names': x_names,
        }

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
