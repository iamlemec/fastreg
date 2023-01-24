##
## extra design stuff, of questionable use rn
##

from operator import and_
from functools import partial

import jax
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_reduce

from .meta import MetaFormula
from .formula import valid_rows, all_valid

# adds in is_leaf option to stock jax version
def tree_transpose(outer_treedef, inner_treedef, pytree_to_transpose, is_leaf=None):
    flat, treedef = tree_flatten(pytree_to_transpose, is_leaf=is_leaf)
    inner_size = inner_treedef.num_leaves
    outer_size = outer_treedef.num_leaves
    if treedef.num_leaves != (inner_size * outer_size):
        expected_treedef = outer_treedef.compose(inner_treedef)
        raise TypeError(f"Mismatch\n{treedef}\n != \n{expected_treedef}")
    iter_flat = iter(flat)
    lol = [[next(iter_flat) for _ in range(inner_size)] for __ in range(outer_size)]
    transposed_lol = zip(*lol)
    subtrees = map(partial(tree_unflatten, outer_treedef), transposed_lol)
    return tree_unflatten(inner_treedef, subtrees)

# tree of formulas -> tree of values and tree of labels
def design_tree(tree, data=None, extern=None, dropna=True, validate=False, valid0=None):
    # turn Formulas into lists of Terms and get structure
    struct = jax.tree_util.tree_structure(tree)

    # use eval to get data, labels, valid
    def eval_term(term):
        args = {'flatten': True} if isinstance(term, MetaFormula) else {'squeeze': True}
        return term.eval(data, extern=extern, method='ordinal', **args)
    info = tree_map(eval_term, tree)

    # unpack into separate trees (hacky)
    struct0 = jax.tree_util.tree_structure((0, 0, 0))
    values, labels, valids = tree_transpose(
        struct, struct0, info,
        is_leaf=lambda x: type(x) is list and set(map(type, x)) == {str}
    )

    # validate all data
    valid = all_valid(valid0, tree_reduce(and_, tree_map(valid_rows, valids)))
    if dropna:
        values = tree_map(lambda x: x[valid], values)

    # return requested info
    if validate:
        return values, labels, valid
    else:
        return values, labels
