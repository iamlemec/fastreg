##
## results summary
##

import numpy as np
import pandas as pd
from scipy.stats.distributions import norm

##
## constants
##

z95 = norm.ppf(0.975)

##
## param summary
##

def param_table(beta, sigma, names):
    # standard errors
    stderr = np.sqrt(sigma.diagonal())

    # confidence interval
    low95 = beta - z95*stderr
    high95 = beta + z95*stderr

    # p-value
    zscore = beta/stderr
    pvalue = 2*(1-norm.cdf(np.abs(zscore)))

    # return all
    return pd.DataFrame({
        'coeff': beta,
        'stderr': stderr,
        'low95': low95,
        'high95': high95,
        'pvalue': pvalue
    }, index=names)
