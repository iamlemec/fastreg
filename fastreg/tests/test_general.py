##
## glm test suite
##

import numpy as np
import fastreg as fr
from fastreg import I, R, C, zero_inflate

# standard formulas

p, p0 = R.p, R.p0
x0 = I + R.x1 + R.x2
x = x0 + C.id1 + C.id2

# helper functions

def frame_valid(df):
    return ~np.isnan(df).any().any()

# tests

def test_glm_real(data_count):
    ret = fr.glm(y=p0, x=x0, data=data_count, loss='poisson')

    assert frame_valid(ret)

def test_glm_categ(data_count):
    ret = fr.glm(y=p, x=x, data=data_count, loss='poisson')

    assert frame_valid(ret)

def test_glm_hdfe(data_count):
    ret = fr.glm(y=p, x=x0+C.id1, data=data_count, loss='poisson', hdfe=C.id2)

    assert frame_valid(ret)

def test_glm_zinf(data_count):
    ret = fr.glm(y=R.pz0, x=x0, data=data_count, loss=zero_inflate('poisson'), extra={'lpzero': 0.0})

    assert frame_valid(ret)

def test_glm_offset(data_count):
    ret = fr.glm(y=R.pt0, x=x0, data=data_count, loss='poisson', offset=R('log(t)'))

    assert frame_valid(ret)
