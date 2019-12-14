##
## regressions
##

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .design import design_matrices, frame_eval, frame_matrix, absorb_categorical
from .summary import param_table

## high dimensional fixed effects
# x expects strings or expressions
# fe can have strings or tuples of strings
def ols(y, x=[], fe=[], data=None, absorb=None, cluster=None, intercept=True, drop='first', output='table'):
    if len(x) == 0 and len(fe) == 0 and not intercept:
        raise(Exception('No columns present!'))

    # make design matrices
    y_vec, x_mat, x_names = design_matrices(y, x=x, fe=fe, data=data, intercept=intercept, drop=drop)
    N, K = x_mat.shape

    # use absorption
    if absorb is not None:
        if cluster is None:
            cluster = absorb
        if type(absorb) is str:
            c_abs = frame_eval(absorb, data)
        else:
            c_abs = frame_matrix(absorb, data)
        y_vec, x_mat, c_idx = absorb_categorical(y_vec, x_mat, c_abs)

    # linalg tool select
    if sp.issparse(x_mat):
        solve = sp.linalg.spsolve
        inv = sp.linalg.inv
    else:
        solve = np.linalg.solve
        inv = np.linalg.inv

    # find point estimates
    xpx = x_mat.T.dot(x_mat)
    xpy = x_mat.T.dot(y_vec)
    beta = solve(xpx, xpy)

    # just the betas
    if output == 'beta':
        return beta

    # find residuals
    y_hat = x_mat.dot(beta)
    e_hat = y_vec - y_hat

    # find standard errors
    ixpx = inv(xpx)
    if cluster not in (None, False):
        xe2 = np.zeros((K, K))
        for v, sel in c_idx.items():
            xei = np.dot(x_mat[sel, :].T, e_hat[sel, None])
            xe2 += np.dot(xei, xei.T)
        sigma = np.dot(np.dot(ixpx, xe2), ixpx)
    else:
        s2 = np.sum(e_hat**2)/(N-K)
        sigma = s2*ixpx

    if output == 'table':
        return param_table(beta, sigma, x_names)
    else:
        return beta, sigma, x_names
