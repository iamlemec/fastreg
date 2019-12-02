##
## regressions
##

import numpy as np
import pandas as pd
from  scipy.stats.distributions import norm
import scipy.sparse as sp
from .design import design_matrices

## high dimensional fixed effects
# x expects strings or expressions
# fe can have strings or tuples of strings
def ols(y, x=[], fe=[], data=None, intercept=True, drop='first'):
    # make design matrices
    y_vec, x_mat, x_names = design_matrices(y, x, fe, data, intercept=intercept, drop=drop)
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
    betas = solve(xpx, xpy)

    # find standard errors
    y_hat = x_mat.dot(betas)
    e_hat = y_vec - y_hat
    s2 = np.sum(e_hat**2)/(N-K)
    cov = s2*inv(xpx)
    stderr = np.sqrt(cov.diagonal())

    # confidence interval
    s95 = norm.ppf(0.975)
    low95 = betas - s95*stderr
    high95 = betas + s95*stderr

    # p-value
    zscore = betas/stderr
    pvalue = 1 - norm.cdf(np.abs(zscore))

    # dataframe of results
    return pd.DataFrame({
        'coeff': betas,
        'stderr': stderr,
        'low95': low95,
        'high95': high95,
        'pvalue': pvalue
    }, index=x_names)
