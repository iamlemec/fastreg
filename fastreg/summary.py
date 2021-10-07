##
## results summary
##

import numpy as np
import pandas as pd
from scipy.stats.distributions import norm

from .tools import maybe_diag

##
## constants
##

z95 = norm.ppf(0.975)

##
## param summary
##

def param_table(beta, sigma, y_name, x_names):
    # standard errors
    if type(sigma) is tuple:
        sigr, sigc = sigma
        stderr = np.sqrt(np.hstack([maybe_diag(sigr), sigc]))
    else:
        stderr = np.sqrt(maybe_diag(sigma))

    # confidence interval
    low95 = beta - z95*stderr
    high95 = beta + z95*stderr

    # p-value
    zscore = beta/stderr
    pvalue = 2*(1-norm.cdf(np.abs(zscore)))

    # return all
    frame = pd.DataFrame({
        'coeff': beta,
        'stderr': stderr,
        'low95': low95,
        'high95': high95,
        'pvalue': pvalue
    }, index=x_names)
    frame = frame.rename_axis(y_name, axis=1)

    return frame
