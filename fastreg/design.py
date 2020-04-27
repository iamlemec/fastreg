##
## design matrices
##

import numpy as np
import scipy.sparse as sp
from patsy.desc import ModelDesc
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from pandas._libs import algos
from itertools import product, chain

##
## vector tools
##

# handles empty data
def vstack(v):
    if len(v) == 0:
        return None
    else:
        return np.vstack(v)

# allows None's and handles empty data
def hstack(v):
    v = [x for x in v if x is not None]
    if len(v) == 0:
        return None
    if any([sp.issparse(x) for x in v]):
        agg = lambda z: sp.hstack(z, format='csr')
    else:
        agg = np.hstack
    return agg(v)

# this assumes row major to align with product
def strides(v):
    if len(v) == 1:
        return np.array([1])
    else:
        return np.r_[1, np.cumprod(v[1:])][::-1]

def swizzle(ks, vs):
    return ','.join([f'{k}={v}' for k, v in zip(ks, vs)])

def chainer(v):
    return list(chain.from_iterable(v))

##
## categorical tools
##

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

# assumes packed integer codes
def reverse_indices(codes):
    ncats = codes.max() + 1
    r, counts = algos.groupsort_indexer(codes, ncats)
    counts = counts.cumsum()
    result = [r[start:end] for start, end in zip(counts, counts[1:])]
    return result

# this is from statsmodels
def group_sums(x, codes):
    if x.ndim == 1:
        return np.bincount(codes, weights=x)
    else:
        return np.vstack([np.bincount(codes, weights=x[:, j]) for j in range(x.shape[1])]).T

def group_means(x, codes):
    if x.ndim == 1:
        return group_sums(x, codes)/np.bincount(codes)
    else:
        return group_sums(x, codes)/np.bincount(codes)[:, None]

##
## design matrices
##

def frame_eval(exp, data, engine='pandas'):
    if engine == 'pandas':
        return data.eval(exp).values
    elif engine == 'python':
        return eval(exp, globals(), data).values

def frame_matrix(terms, data):
    if type(terms) is str:
        terms = [terms]
    return vstack([frame_eval(z, data) for z in terms]).T

# this is mildly inefficient in the case of overlap
def encode_categorical(terms, data, method='sparse', drop='first'):
    if (K := len(terms)) == 0:
        if method == 'ordinal':
            return None, {}
        elif method == 'sparse':
            return None, []

    # ordinal encode each term
    terms = [(t,) if type(t) is str else t for t in terms]
    term_mats = [frame_matrix(t, data) for t in terms]
    term_vals, term_labels = zip(*[category_indices(m, return_labels=True) for m in term_mats])
    form_mat = vstack(term_vals).T

    # compress indices with chosen method
    # its assumed that if you're doing ordinal, you don't need labels to be dropped or flattened
    if method == 'ordinal':
        enc = OrdinalEncoder(categories='auto', dtype=np.int)
        form_enc = enc.fit_transform(form_mat)
        cats_all = enc.categories_
        seen_labels = [[n[i] for i in c] for c, n in zip(cats_all, term_labels)]
        form_labels = {':'.join(k): v for k, v in zip(terms, seen_labels)}
    elif method == 'sparse':
        enc = OneHotEncoder(categories='auto', drop=drop, dtype=np.int)
        form_enc = enc.fit_transform(form_mat)
        drop_idx = enc.drop_idx_
        cats_all = enc.categories_

        # find all cross-term names
        if drop_idx is None:
            seen_cats = cats_all
        else:
            seen_cats = [np.delete(c, i) for c, i in zip(cats_all, drop_idx)]
        seen_labels = [[n[i] for i in c] for c, n in zip(seen_cats, term_labels)]
        form_labels = chainer([swizzle(t, i) for i in n] for t, n in zip(terms, seen_labels))

    return form_enc, form_labels

##
## categorical absorption
##

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

def parse_term(term):
    return tuple(f.code for f in term.factors)

def classify_term(term):
    cats = [f.startswith('C(') and f.endswith(')') for f in term]
    if len(term) == 0:
        return 'intercept'
    elif all(cats):
        return 'categorical'
    elif all([not c for c in cats]):
        return 'continuous'
    else:
        raise(Exception(f'Can\'t mix continuous and categorical right now: {term}'))

def strip_cat(term):
    return tuple(f[2:-1] for f in term)

def squeeze_term(term):
    return term[0] if len(term) == 1 else term

# this obviously can't handle anything but treatment coding, but that's required for sparsity
def parse_formula(form):
    # use patsy for formula parse
    desc = ModelDesc.from_formula(form)

    # convert to string lists
    y_terms = [parse_term(t) for t in desc.lhs_termlist]
    x_terms = [parse_term(t) for t in desc.rhs_termlist]
    x_class = [classify_term(t) for t in x_terms]

    # separate into components
    y = squeeze_term(y_terms[0])
    x = [squeeze_term(t) for t, c in zip(x_terms, x_class) if c == 'continuous']
    fe = [squeeze_term(strip_cat(t)) for t, c in zip(x_terms, x_class) if c == 'categorical']
    intercept = any([c == 'intercept' for c in x_class])

    return y, x, fe, intercept

##
## design interface
##

def design_matrix(x=[], fe=[], data=None, intercept=True, method='sparse', drop='first', output=None):
    # construct individual matrices
    x_mat, x_names = frame_matrix(x, data), x
    c_mat, c_labels = encode_categorical(fe, data, method=method, drop=drop)

    # get data length
    N = len(data)

    # optionally add intercept
    if intercept:
        inter = np.ones((N, 1))
        x_mat = np.hstack([inter, x_mat]) if x_mat is not None else inter
        x_names = ['one'] + x_names

    return x_mat, x_names, c_mat, c_labels

def design_matrices(y, x=[], fe=[], formula=None, data=None, intercept=True, method='sparse', drop='first', output=None):
    if formula is not None:
        y, x, fe, intercept = parse_formula(formula)
    y_vec = frame_eval(y, data)
    x_mat, x_names, c_mat, c_labels = design_matrix(x=x, fe=fe, data=data, intercept=intercept, method=method, drop=drop, output=output)
    return y_vec, x_mat, x_names, c_mat, c_labels
