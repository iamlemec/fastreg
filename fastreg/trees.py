##
## extra design stuff, of questionable use rn
##

from operator import and_
from functools import partial

import jax
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_reduce

from .meta import MetaFormula
from .formula import valid_rows, all_valid

# modified from stock jax version to allow variable inner shapes
def tree_transpose(outer_treedef, inner_treedef, pytree_to_transpose):
    # get full tree structure and flatten up to
    full_treedef = outer_treedef.compose(inner_treedef)
    flat = full_treedef.flatten_up_to(pytree_to_transpose)

    # get inner and outer sizes
    inner_size = inner_treedef.num_leaves
    outer_size = outer_treedef.num_leaves

    # create nested lists and transpose
    iter_flat = iter(flat)
    lol = [[next(iter_flat) for _ in range(inner_size)] for __ in range(outer_size)]
    transposed_lol = zip(*lol)

    # remap into desired tree structure
    subtrees = map(partial(tree_unflatten, outer_treedef), transposed_lol)
    return tree_unflatten(inner_treedef, subtrees)

# tree of formulas -> tree of values and tree of labels
def design_tree(tree, data=None, extern=None, dropna=True, validate=False, valid0=None):
    # use eval to get data, labels, valid
    def eval_item(term):
        args = {'flatten': True} if isinstance(term, MetaFormula) else {'squeeze': True}
        return term.eval(data, extern=extern, method='ordinal', **args)
    info = tree_map(eval_item, tree)

    # unpack into separate trees (hacky)
    struct_outer = jax.tree_util.tree_structure(tree)
    struct_inner = jax.tree_util.tree_structure((0, 0, 0))
    values, labels, valids = tree_transpose(struct_outer, struct_inner, info)

    # validate all data
    valid = all_valid(valid0, tree_reduce(and_, tree_map(valid_rows, valids)))
    if dropna:
        values = tree_map(lambda x: x[valid], values)

    # return requested info
    if validate:
        return values, labels, valid
    else:
        return values, labels
