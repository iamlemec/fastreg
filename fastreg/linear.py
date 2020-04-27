##
## regressions
##

import numpy as np
import scipy.sparse as sp

from .design import design_matrices, frame_matrix, absorb_categorical, category_indices, group_sums, hstack
from .summary import param_table

## high dimensional fixed effects
# x expects strings or expressions
# fe can have strings or tuples of strings
def ols(y=None, x=[], fe=[], formula=None, data=None, absorb=None, cluster=None, intercept=True, drop='first', output='table', method='solve'):
    # make design matrices
    y_vec, x0_mat, x0_names, c_mat, c_names = design_matrices(y, x=x, fe=fe, formula=formula, data=data, intercept=intercept, drop=drop)
    N, K = x0_mat.shape

    # combine x variables
    x_mat = hstack([x0_mat, c_mat])
    x_names = x0_names + c_names

    # use absorption
    if absorb is not None:
        cluster = absorb
        c_abs = frame_matrix(absorb, data)
        y_vec, x_mat, keep = absorb_categorical(y_vec, x_mat, c_abs)

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
        c_mat = frame_matrix(cluster, data)
        if absorb is not None:
            c_mat = c_mat[keep, :]

        # from cameron and miller
        codes = category_indices(c_mat)
        xeg = group_sums(x_mat*e_hat[:, None], codes)
        xe2 = xeg.T @ xeg
        sigma = ixpx @ xe2 @ ixpx
    else:
        s2 = (e_hat @ e_hat)/(N-K)
        sigma = s2*ixpx

    if output == 'table':
        return param_table(beta, sigma, x_names)
    else:
        return {
            'beta': beta,
            'sigma': sigma,
            'x_names': x_names,
            'y_hat': y_hat,
            'e_hat': e_hat
        }
