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

def param_table(beta, sigma, y_name, x_names):
    # standard errors
    stderr = np.sqrt(sigma.diagonal())

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
