##
## ols test suite
##

import re
import pytest
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import fastreg as fr
from fastreg import I, R, C, NONE

# standard formulas

y, y0 = R.y, R.y0
x0 = I + R.x1 + R.x2
x = x0 + C.id1 + C.id2

# helper functions

def statsmodels_rename(s):
    if s == 'Intercept':
        return 'I'
    elif ret := re.match(r'C\(([^\)]+)\)\[T\.([^\]]+)\]', s):
        return '='.join(ret.groups())
    else:
        return s

def statsmodels_frame(res):
    df = pd.DataFrame({
        'coeff': res.params,
        'stderr': np.sqrt(np.diagonal(res.cov_params()))
    })
    df = df.rename_axis(res.model.endog_names, axis=1)
    df = df.rename(statsmodels_rename, axis=0)
    return df

def statsmodels_isclose(fit_fr, fit_sm):
    ret_fr = fit_fr[['coeff', 'stderr']].sort_index()
    ret_sm = statsmodels_frame(fit_sm).sort_index()
    return np.isclose(ret_fr, ret_sm).all()

def frame_valid(df):
    return ~np.isnan(df).any().any()

# tests

def test_ols_real(data):
    ret = fr.ols(y=y0, x=x0, data=data)

    assert frame_valid(ret)

def test_ols_categ(data):
    ret = fr.ols(y=y, x=x, data=data)

    assert frame_valid(ret)

def test_ols_real_statsmodels(data):
    fit_fr = fr.ols(y=y0, x=x0, data=data)
    fit_sm = smf.ols('y0 ~ 1 + x1 + x2', data=data).fit()

    assert statsmodels_isclose(fit_fr, fit_sm)

def test_ols_categ_statsmodels(data):
    xp = R.x1 + R.x2 + C.id1(NONE) + C.id2
    fit0_fr = fr.ols(y=y, x=xp, data=data)
    fit0_sm = smf.ols(
        'y ~ 0 + x1 + x2 + C(id1) + C(id2)', data=data, eval_env=-1
    ).fit()

    fit1_fr = fr.ols(y=y, x=x, data=data)
    fit1_sm = smf.ols(
        'y ~ 1 + x1 + x2 + C(id1) + C(id2)', data=data, eval_env=-1
    ).fit()

    assert statsmodels_isclose(fit0_fr, fit0_sm)
    assert statsmodels_isclose(fit1_fr, fit1_sm)

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

@pytest.mark.parametrize('cov', [0, 1, 2, 3])
def test_ols_robust(data, cov):
    ret = fr.ols(y=y0, x=x0, data=data, stderr=f'hc{cov}')

    assert frame_valid(ret)

@pytest.mark.parametrize('cov', [0, 1, 2, 3])
def test_ols_robust_statsmodels(data, cov):
    fit_fr = fr.ols(y=y0, x=x0, data=data, stderr=f'hc{cov}')
    fit_sm = smf.ols('y0 ~ 1 + x1 + x2', data=data).fit(cov_type=f'HC{cov}')

    assert statsmodels_isclose(fit_fr, fit_sm)
