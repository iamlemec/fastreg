##
## design matrices
##

import re
import numpy as np
from patsy.desc import ModelDesc
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from itertools import product

from .tools import hstack, vstack, group_means, strides, chainer, categorize

##
## design matrices
##

def term_eval(term, data, prod=True):
    if isinstance(term, str):
        term = term,
    if len(term) == 0:
        vals = np.ones(len(data))
        return vals if prod else vals.reshape(-1, 1)
    elif len(term) == 1:
        vals = data.eval(term[0]).values
        return vals if prod else vals.reshape(-1, 1)
    else:
        vals = data.eval(term).T
        return vals.prod(axis=1) if prod else vals

def term_name(term, ident='one'):
    if isinstance(term, str):
        term = term,
    if len(term) == 0:
        return ident
    else:
        return ':'.join(term)

def formula_eval(terms, data):
    if len(terms) == 0:
        return None, []
    else:
        mat = vstack([term_eval(t, data, prod=True) for t in terms]).T
        nam = [term_name(t) for t in terms]
        return mat, nam

##
## categorical tools
##

# make labels
def swizzle(ks, vs):
    return ','.join([f'{k}={v}' for k, v in zip(ks, vs)])

# ordinally encode interactions terms (tuple-like things)
def category_indices(vals, return_labels=False):
    if vals.ndim == 1:
        vals = vals[:, None]

    # convert to packed integers
    ord_enc = OrdinalEncoder(categories='auto', dtype=np.int)
    ord_vals = ord_enc.fit_transform(vals)
    ord_cats = ord_enc.categories_

    # interact with product
    ord_sizes = [len(x) for x in ord_cats]
    ord_strides = strides(ord_sizes)
    ord_cross = ord_vals @ ord_strides

    # return requested
    if return_labels:
        ord_labels = list(product(*ord_cats))
        return ord_cross, ord_labels
    else:
        return ord_cross

# this is mildly inefficient in the case of overlap
def encode_categorical(facts, data, method='sparse', drop='first'):
    # ensure tuple
    if isinstance(facts, str):
        facts = facts,

    # separate real and categorical
    categ, reals = categorize(is_categorical, facts)

    # get categoricals
    categ_mat = term_eval(categ, data, prod=False)
    categ_vals, categ_labels = category_indices(categ_mat, return_labels=True)
    categ_vals = categ_vals.reshape(-1, 1)
    categ_labels = [swizzle(categ, l) for l in categ_labels]

    # compress indices with chosen method
    # if ordinal no labels are dropped
    if method == 'ordinal':
        enc = OrdinalEncoder(categories='auto', dtype=int)
        cats_enc = enc.fit_transform(categ_vals)
        cats_all, = enc.categories_
    elif method == 'sparse':
        enc = OneHotEncoder(categories='auto', drop=drop, dtype=int)
        cats_enc = enc.fit_transform(categ_vals)
        cats_all, = enc.categories_
        if enc.drop_idx_ is not None:
            cats_all = np.delete(cats_all, enc.drop_idx_[0])

    # handle reals
    if len(reals) > 0:
        reals_vec = term_eval(reals, data, prod=True)
        reals_name = term_name(reals)
        cats_enc = cats_enc.multiply(reals_vec.reshape(-1, 1))
        categ_labels = [f'{l}:{reals_name}' for l in categ_labels]

    # get used lables
    cats_label = [categ_labels[i] for i in cats_all]

    return cats_enc, cats_label

def absorb_categorical(y, x, abs):
    N, K = x.shape
    _, A = abs.shape

    # copy so as not to destroy
    y = y.copy()
    x = x.copy()

    # store original means
    avg_y0 = np.mean(y)
    avg_x0 = np.mean(x, axis=0)

    # track whether to drop
    keep = np.ones(N, dtype=np.bool)

    # do this iteratively to reduce data loss
    for j in range(A):
        # create class groups
        codes = category_indices(abs[:, j])

        # perform differencing on y
        avg_y = group_means(y, codes)
        y -= avg_y[codes]

        # perform differencing on x
        avg_x = group_means(x, codes)
        x -= avg_x[codes, :]

        # detect singletons
        multi = np.bincount(codes) > 1
        keep &= multi[codes]

    # recenter means
    y += avg_y0
    x += avg_x0[None, :]

    # drop singletons
    y = y[keep]
    x = x[keep, :]

    return y, x, keep

##
## R style formulas
##

# real
class R(str):
    pass

# categorical
class C(str):
    pass

# evaluator
class I(str):
    pass

# lookup table
FTYPES = {'R': R, 'C': C, 'I': I}

def parse_factor(fact):
    ret = re.match(r'(R|C|I)\(([^\)]+)\)', fact.code)
    if ret is not None:
        pre, name = ret.groups()
        return FTYPES[pre](name)
    else:
        return fact.code

def parse_term(term):
    facts = tuple(parse_factor(f) for f in term.factors)
    return squeeze_term(facts)

def squeeze_term(term):
    return term[0] if len(term) == 1 else term

def is_categorical(term):
    if isinstance(term, str):
        term = term,
    return any([type(f) is C for f in term])

# this can only handle treatment coding, but that's required for sparsity
def parse_formula(form):
    # use patsy for formula parse
    desc = ModelDesc.from_formula(form)
    lhs, rhs = desc.lhs_termlist, desc.rhs_termlist

    # convert to string lists
    y_terms = parse_term(lhs[0]) if len(lhs) > 0 else None
    x_terms = [parse_term(t) for t in rhs]

    return y_terms, x_terms

##
## design interface
##

def design_matrix(
    terms=[], data=None, intercept=True, method='sparse', drop='first',
    output=None
):
    # ensure intercept
    if intercept and () not in terms:
        terms.insert(0, ())

    # separate pure real and categorical
    c_terms, x_terms = categorize(is_categorical, terms)

    # compute real matrices
    if len(x_terms) > 0:
        x_mat, x_names = formula_eval(x_terms, data)
    else:
        x_mat, x_names = None, []

    # compute categorical matices
    if len(c_terms) > 0:
        c_mats, c_labels = zip(*[
            encode_categorical(ct, data, method=method, drop=drop) for ct in c_terms
        ])
        c_mat = hstack(c_mats)
    else:
        c_mat, c_labels = None, []

    # get combined category names
    if method == 'sparse':
        c_names = chainer(c_labels)
    elif method == 'ordinal':
        c_names = {t: ls for t, ls in zip(c_terms, c_labels)}

    return x_mat, x_names, c_mat, c_names

def design_matrices(
    y, x=[], formula=None, data=None, intercept=True, method='sparse',
    drop='first', output=None
):
    if formula is not None:
        y, x = parse_formula(formula)

    y_vec = term_eval(y, data)
    x_mat, x_names, c_mat, c_labels = design_matrix(
        terms=x, data=data, intercept=intercept, method=method, drop=drop,
        output=output
    )

    return y_vec, x_mat, x_names, c_mat, c_labels
