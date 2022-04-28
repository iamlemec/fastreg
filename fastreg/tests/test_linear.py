##
## fastreg test suite
##

import re
import pytest
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import fastreg as fr
from fastreg import I, R, C

# standard formulas

y, y0 = R.y, R.y0
x0 = I + R.x1 + R.x2
x = x0 + C.id1 + C.id2

# helper functions

def rename_statsmodels(s):
    if s == 'Intercept':
        return 'I'
    elif ret := re.match(r'C\(([^\)]+)\)\[T\.([^\]]+)\]', s):
        return '='.join(ret.groups())
    else:
        return s

# tests

def test_ols_real(data):
    ret = fr.ols(y=y0, x=x0, data=data)

    assert ~np.isnan(ret).any().any()

def test_ols_categ(data):
    ret = fr.ols(y=y, x=x, data=data)

    assert ~np.isnan(ret).any().any()

def test_ols_real_statsmodels(data):
    fit_fr = fr.ols(y=y0, x=x0, data=data)
    fit_sm = smf.ols('y0 ~ 1 + x1 + x2', data=data).fit()

    ret_fr = fit_fr[['coeff', 'stderr']].sort_index()
    ret_sm = pd.DataFrame({
        'coeff': fit_sm.params,
        'stderr': np.sqrt(np.diagonal(fit_sm.cov_params()))
    })
    ret_sm = ret_sm.rename_axis('y0', axis=1)
    ret_sm = ret_sm.rename(rename_statsmodels, axis=0).sort_index()

    assert np.isclose(ret_fr, ret_sm).all()

def test_ols_categ_statsmodels(data):
    fit_fr = fr.ols(y=y, x=x, data=data)
    fit_sm = smf.ols(
        'y ~ 1 + x1 + x2 + C(id1) + C(id2)', data=data, eval_env=-1
    ).fit()

    ret_fr = fit_fr[['coeff', 'stderr']].sort_index()
    ret_sm = pd.DataFrame({
        'coeff': fit_sm.params,
        'stderr': np.sqrt(np.diagonal(fit_sm.cov_params()))
    })
    ret_sm = ret_sm.rename_axis('y0', axis=1)
    ret_sm = ret_sm.rename(rename_statsmodels, axis=0).sort_index()

    assert np.isclose(ret_fr, ret_sm).all()

def test_absorb(data):
    ret0 = fr.ols(y=y, x=x, data=data)
    ret1 = fr.ols(y=y, x=x, absorb=C.id1, data=data)
    ret2 = fr.ols(y=y, x=x, absorb=C.id2, data=data)

    elreg1 = r'(x1|x2|id2=.+)'
    cx01 = ret0.filter(regex=elreg1, axis=0)['coeff']
    cx11 = ret1.filter(regex=elreg1, axis=0)['coeff']

    elreg2 = r'(x1|x2|id1=.+)'
    cx02 = ret0.filter(regex=elreg2, axis=0)['coeff']
    cx22 = ret2.filter(regex=elreg2, axis=0)['coeff']

    assert np.isclose(cx01, cx11).all()
    assert np.isclose(cx02, cx22).all()

def test_hdfe(data):
    ret0 = fr.ols(y=y, x=x, data=data).sort_index()
    ret1 = fr.ols(y=y, x=x, hdfe=C.id1, data=data).sort_index()
    ret2 = fr.ols(y=y, x=x, hdfe=C.id2, data=data).sort_index()

    assert np.isclose(ret0, ret1).all()
    assert np.isclose(ret0, ret2).all()
