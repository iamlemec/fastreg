##
## extra design stuff, of questionable use rn
##

from operator import and_
from functools import partial

import jax.numpy as np  
import jax
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_reduce

from .meta import MetaFormula
from .formula import valid_rows, all_valid, is_categorical, prune_categories

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
    valid0=None, prune=True, warn=False
):
    # use eval to get data, labels, valid
    def eval_item(term):
        args = {'flatten': True} if isinstance(term, MetaFormula) else {'squeeze': True}
        return term.eval(data, extern=extern, method=method, **args)
    info = tree_map(eval_item, tree)

    # unpack into separate trees (hacky)
    struct_outer = jax.tree_util.tree_structure(tree)
    struct_inner3 = jax.tree_util.tree_structure((0, 0, 0))
    values, labels, valids = tree_transpose(struct_outer, struct_inner3, info)
    valid = all_valid(valid0, tree_reduce(and_, valids))

    # drop data and prune cats if requested
    if dropna:
        values = tree_map(lambda x: x[valid], values)

    # prune categories for pure categoricals
    if prune:
        def prune_cats(i, v, l):
            print(is_categorical(i))
            if is_categorical(i, strict=True):
                return prune_categories(v, l, method=method, warn=warn)
            else:
                return v, l
        pruned = tree_map(prune_cats, tree, values, labels)
        struct_inner2 = jax.tree_util.tree_structure((0, 0))
        values, labels = tree_transpose(struct_outer, struct_inner2, pruned)

    # return requested info
    if validate:
        return values, labels, valid
    else:
        return values, labels
