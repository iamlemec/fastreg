import pytest
import numpy as np
import statsmodels.formula.api as smf
import fastreg as fr
from fastreg import I, R, C

def test_ols(data):
    ret0 = fr.ols(y=R.y0, x=I+R.x1+R.x2, data=data)
    ret1 = fr.ols(y=R.y, x=I+R.x1+R.x2+C.id1+C.id2, data=data)

def test_statsmodels(data):
    par_fr = fr.ols(y=R.y0, x=I+R.x1+R.x2, data=data)['coeff']
    par_sm = smf.ols('y0 ~ 1 + x1 + x2', data=data).fit().params

    assert np.isclose(par_fr, par_sm).all()

def test_absorb(data):
    ret1 = fr.ols(y=R.y, x=I+R.x1+R.x2+C.id1+C.id2, data=data)
    ret2 = fr.ols(y=R.y, x=I+R.x1+R.x2+C.id1, absorb=C.id2, data=data)

    elreg = r'(x1|x2|id1=.+)'
    cx1 = ret1.filter(regex=elreg, axis=0)['coeff']
    cx2 = ret1.filter(regex=elreg, axis=0)['coeff']

    assert np.isclose(cx1, cx2).all()
