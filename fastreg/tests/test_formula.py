##
## formula test suite
##

import re
import pytest
import fastreg as fr
from fastreg import O, I, R, C

def test_factor():
    assert I == I
    assert I != R.x

    assert R.x == R.x
    assert R.x != R.y

def test_term():
    assert I*R.x == R.x
    assert I*R.x != R.y

    assert R.x*R.x == R.x*R.x
    assert R.x*R.x != R.x*R.y

    assert R.x*R.y == R.x*R.y
    assert R.x*R.y != R.x*R.z

    assert R.x*R.y == R.y*R.x
    assert R.x*R.y != R.x*R.z

def test_formula():
    assert R.x + R.y == R.x + R.y
    assert R.x + R.y != R.x + R.z

    assert R.x + R.y == R.y + R.x
    assert R.x + R.y != R.y + R.z

    assert R.x*(R.y+R.z) == R.x*R.y + R.x*R.z
    assert R.x*(R.y+R.z) != R.x*R.y + R.x*R.w

def test_addsub():
    assert I + I == I
    assert I + R.x != I

    assert R.x + R.x == R.x
    assert R.x + R.y != R.x

    assert I - I == O
    assert I - R.x != O

    assert (R.x + R.y) - R.y == R.x
    assert (R.x + R.y) - R.z != R.x
