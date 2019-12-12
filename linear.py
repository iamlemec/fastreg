##
## regressions
##

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .design import design_matrices
from .summary import param_table

## high dimensional fixed effects
# x expects strings or expressions
# fe can have strings or tuples of strings
def ols(y, x=[], fe=[], data=None, absorb=None, intercept=True, drop='first'):
    if len(x) == 0 and len(fe) == 0 and not intercept:
        raise(Exception('No columns present!'))

    # make design matrices
    y_vec, x_mat, x_names, c_abs = design_matrices(y, x=x, fe=fe, data=data, absorb=absorb, intercept=intercept, drop=drop)
    N, K = x_mat.shape

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

    # find residuals
    y_hat = x_mat.dot(beta)
    e_hat = y_vec - y_hat

    # find standard errors
    ixpx = inv(xpx)
    if absorb is not None:
        # create class groups
        vals = pd.Categorical(c_abs)
        group = vals._reverse_indexer()

        # aggregate by class
        xe2 = np.zeros((K, K))
        for v, sel in group.items():
            xei = np.dot(x_mat[sel, :].T, e_hat[sel, None])
            xe2 += np.dot(xei, xei.T)
        sigma = np.dot(np.dot(ixpx, xe2), ixpx)
    else:
        s2 = np.sum(e_hat**2)/(N-K)
        sigma = s2*ixpx

    return param_table(beta, sigma, x_names)
