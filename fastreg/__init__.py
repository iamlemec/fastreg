from . import tools
from . import formula
from . import linear
from . import testing

from .tools import (
    valid_rows, all_valid
)
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
    import jax
    HAS_JAX = True
except:
    HAS_JAX = False

if HAS_JAX:
    from . import general
    from . import trees

    from .general import (
        glm, logit, poisson, negbin, poisson_zinf, negbin_zinf, gols,
        zero_inflate, add_offset, losses, binary_loss, poisson_loss,
        negbin_loss, lstsq_loss, normal_loss, maxlike, maxlike_panel,
        glm_model, adam, optax_wrap
    )
    from .trees import design_tree
