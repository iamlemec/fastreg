##
## regressions
##

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .design import design_matrices, frame_eval, frame_matrix, absorb_categorical, category_indices
from .summary import param_table

## high dimensional fixed effects
# x expects strings or expressions
# fe can have strings or tuples of strings
def ols(y, x=[], fe=[], data=None, absorb=None, cluster=None, intercept=True, drop='first', output='table'):
    # make design matrices
    y_vec, x_mat, x_names = design_matrices(y, x=x, fe=fe, data=data, intercept=intercept, drop=drop)
    N, K = x_mat.shape

    # use absorption
    if absorb is not None:
        cluster = absorb
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
    ixpx = inv(xpx)
    beta = ixpx.dot(xpy)

    # just the betas
    if output == 'beta':
        return beta

    # find residuals
    y_hat = x_mat.dot(beta)
    e_hat = y_vec - y_hat

    # find standard errors

    if cluster is not None:
        # if we haven't already calculated for absorb
        if absorb is None:
            cluster = frame_matrix(cluster, data)
            _, c_idx = category_indices(cluster)

        # from cameron and miller
        xe2 = np.zeros((K, K))
        for sel in c_idx:
            xei = x_mat[sel, :].T.dot(e_hat[sel])
            xe2 += np.outer(xei, xei)
        sigma = np.dot(np.dot(ixpx, xe2), ixpx)
    else:
        s2 = np.dot(e_hat, e_hat)/(N-K)
        sigma = s2*ixpx

    if output == 'table':
        return param_table(beta, sigma, x_names)
    else:
        return beta, sigma, x_names
