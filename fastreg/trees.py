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

# tree of terms -> tree of values and tree of labels
def design_tree(
    tree, data=None, extern=None, method='ordinal', dropna=True, validate=False,
    valid0=None, prune=True, flatten=False, warn=False
):
    # use eval to get data, labels, valid
    info = tree_map(
        lambda x: x.eval(data, extern=extern, method=method, squeeze=True), tree
    )

    # unpack into separate trees (hacky)
    struct_outer = jax.tree_util.tree_structure(tree)
    struct_inner3 = jax.tree_util.tree_structure((0, 0, 0))
    values, labels, valids = tree_transpose(struct_outer, struct_inner3, info)

    # drop invalid rows
    valid = all_valid(valid0, tree_reduce(and_, valids))
    if dropna:
        values = tree_map(lambda x: x[valid], values)

    # prune categories
    if prune:
        prune_cats = lambda v, l: prune_categories(v, l, method=method, warn=False)
        pruned = tree_map(
            lambda i, v, l: prune_cats(v, {i: l}) if is_categorical(i) else (v, l),
            tree, values, labels
        )
        struct_inner2 = jax.tree_util.tree_structure((0, 0))
        values, labels = tree_transpose(struct_outer, struct_inner2, pruned)

    # return requested info
    if validate:
        return values, labels, valid
    else:
        return values, labels
