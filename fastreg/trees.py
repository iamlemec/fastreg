##
## extra design stuff, of questionable use rn
##

from operator import and_
from functools import partial

import jax
import jax.numpy as np
from jax.tree_util import tree_unflatten, tree_map, tree_reduce

from .meta import MetaFactor, MetaTerm, MetaFormula
from .formula import all_valid, is_categorical, prune_categories

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

# subset data allowing for missing chunks
def tree_drop_invalid(values, valid, warn=False):
    V, N = np.sum(valid), len(valid)
    dropper = lambda x: x[valid] if x is not None else x
    if V < N:
        if warn:
            print(f'dropping {N-V}/{N} null rows')
        return tree_map(dropper, values)
    else:
        return values

# tree of terms -> tree of values and tree of labels
def design_tree(
    tree, data=None, extern=None, encoding='ordinal', dropna=True, validate=False,
    valid0=None, prune=True, flatten=True
):
    # use eval to get labels, values, valid
    def eval_term(term):
        col = term.eval(data, extern=extern, encoding=encoding)
        if isinstance(term, (MetaFactor, MetaTerm)):
            labels = col.labels
            values = col.values
            valid = col.valid
        elif isinstance(term, MetaFormula):
            if is_categorical(term):
                (_, labels), (_, values), valid = col
            else:
                (labels, _), (values, _), valid = col
        return labels, values, valid
    spec = tree_map(eval_term, tree)

    # unpack into separate trees (hacky)
    struct_outer = jax.tree_util.tree_structure(tree)
    struct_inner3 = jax.tree_util.tree_structure((0, 0, 0))
    labels, values, valids = tree_transpose(struct_outer, struct_inner3, spec)

    # drop invalid rows
    valid = all_valid(valid0, tree_reduce(and_, valids))
    if dropna:
        values = tree_drop_invalid(values, valid)

    # prune categories
    if prune:
        def prune_cats(t, l, v):
            if type(l) is dict:
                l, v = prune_categories(l, v, encoding=encoding, warn=False)
            return l, v
        pruned = tree_map(prune_cats, tree, labels, values)
        struct_inner2 = jax.tree_util.tree_structure((0, 0))
        labels, values = tree_transpose(struct_outer, struct_inner2, pruned)

    # return requested info
    if validate:
        return labels, values, valid
    else:
        return labels, values
