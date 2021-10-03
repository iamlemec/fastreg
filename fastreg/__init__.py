from . import formula
from . import linear
from . import testing

from .formula import Factor, Term, Formula, design_matrices
from .formula import I, R, C, D, B
from .linear import ols
from .testing import dataset

try:
    from .general import (
        glm, logit, poisson, negbin, zinf_poisson, zinf_negbin, gols
    )
except:
    pass
