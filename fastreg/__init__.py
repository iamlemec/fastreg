from . import tools
from . import formula
from . import linear
from . import testing

from .formula import (
    Factor, Term, Formula, Real, Categ, Demean, Binned, Custom,
    robust_eval, factor, design_matrix, design_matrices,
    O, I, R, C, D, B, C0, B0, Drop, sum0 as sum, drop_type
)
from .linear import ols
from .testing import dataset

NONE = Drop.NONE
FIRST = Drop.FIRST
VALUE = Drop.VALUE

try:
    from .general import (
        glm, logit, poisson, negbin, zinf_poisson, zinf_negbin, gols
    )
except:
    pass
