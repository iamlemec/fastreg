##
## formula test suite
##

import re
import pytest
import fastreg as fr
from fastreg import O, I, R, C, FIRST, NONE, VALUE, Factor, Term, Formula, drop_type

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

def test_droptype():
    assert drop_type(FIRST) == FIRST
    assert drop_type(NONE) == NONE
    assert drop_type(42) == VALUE
    assert drop_type(None) == VALUE

def test_drop():
    assert C.id._drop == FIRST
    assert C.id(FIRST)._drop == FIRST
    assert C.id(NONE)._drop == NONE
    assert C.id(42)._drop == 42
    assert C.id(None)._drop is None

def test_drop_multi():
    assert C.id1.to_term().drop(FIRST)._drop == FIRST
    assert C.id1.to_term().drop(NONE)._drop == NONE
    assert C.id1.to_term().drop(42)._drop == (42,)
    assert C.id1.to_term().drop(None)._drop == (None,)

    assert (C.id1*C.id2)._drop == FIRST
    assert (C.id1('A')*C.id2(1))._drop == ('A', 1)
    assert (C.id1('A')*C.id2(None))._drop == ('A', None)
    assert (C.id1*C.id2).drop('A', 1)._drop == ('A', 1)
    assert (C.id1*C.id2).drop(NONE)._drop == NONE
    assert (C.id1*C.id2).drop('A', None)._drop == ('A', None)
    assert Term(C.id1, C.id2)._drop == FIRST
    assert Term(C.id1, C.id2, drop=NONE)._drop == NONE
    assert Term(C.id1, C.id2, drop=('A', 1))._drop == ('A', 1)
    assert Term(C.id1, C.id2, drop=['A', 1])._drop == ('A', 1)

    assert (R.x1*R.x2)._drop == NONE
    assert (R.x*C.id)._drop == NONE
    assert (C.id1('A')*C.id2)._drop == FIRST
    assert (C.id1(FIRST)*C.id2(NONE))._drop == FIRST
    assert Term(C.id1('A'), C.id2(1), drop=FIRST)._drop == FIRST
    assert Term(C.id1('A'), C.id2, drop=NONE)._drop == NONE
    assert (C.id1('A')*C.id2(None))._drop == ('A', None)

    assert (C.id1*C.id2*C.id3)._drop == FIRST
    assert (C.id1('A')*C.id2(1)*C.id3)._drop == FIRST
    assert (C.id1('A')*C.id2(1)*C.id3(1.5))._drop == ('A', 1, 1.5)
    assert (C.id1('A')*C.id2(1)*C.id3(None))._drop == ('A', 1, None)
    assert (C.id1*C.id2*C.id3).drop('A', 1, 1.5)._drop == ('A', 1, 1.5)
