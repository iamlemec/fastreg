##
## extra design stuff, of questionable use rn
##

from operator import and_
from functools import partial

import jax.numpy as np  
import jax
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_reduce

from .meta import MetaFormula
from .tools import hstack, chainer
from .formula import valid_rows, all_valid, drop_invalid, is_categorical, prune_categories

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
def design_tree(
    tree, data=None, extern=None, method='ordinal', dropna=True, validate=False,
    valid0=None, prune=True, flatten=False, warn=False
):
    # use eval to get data, labels, valid
    def eval_item(term):
        args = {} if isinstance(term, MetaFormula) else {'squeeze': True}
        return term.eval(data, extern=extern, method=method, **args)
    info = tree_map(eval_item, tree)

    # unpack into separate trees (hacky)
    struct_outer = jax.tree_util.tree_structure(tree)
    struct_inner3 = jax.tree_util.tree_structure((0, 0, 0))
    values, labels, valids = tree_transpose(struct_outer, struct_inner3, info)
    valid = all_valid(valid0, tree_reduce(and_, valids))

    # dropna, prune categories, and flatten
    def process(i, v, l):
        if type(v) is tuple and len(v) == 2:
            (vx, vc), (lx, lc) = v, l
        elif is_categorical(i):
            vx, vc = None, v
            lx, lc = None, l
        else:
            vx, vc = v, None
            lx, lc = l, None
        if dropna:
            vx, vc = drop_invalid(valid, vx, vc, warn=False)
        if prune and vc is not None:
            vc, vl = prune_categories(vc, lc, method=method, warn=False)
        if flatten:
            v = hstack([vx, vc])
            l = lx + chainer(lc.values())
        else:
            v, l = (vx, vc), (lx, lc)
        return v, l
    pruned = tree_map(process, tree, values, labels)
    struct_inner2 = jax.tree_util.tree_structure((0, 0))
    values, labels = tree_transpose(struct_outer, struct_inner2, pruned)

    # return requested info
    if validate:
        return values, labels, valid
    else:
        return values, labels
