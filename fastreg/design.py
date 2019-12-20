##
## design matrices
##

import numpy as np
import pandas as pd
import scipy.sparse as sp
from patsy.desc import ModelDesc
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from itertools import product, chain

##
## tools
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
## design
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

def sparse_categorical(terms, data, drop='first'):
    if len(terms) == 0:
        return None, []

    # generate map between terms and features
    terms = [(z,) if type(z) is str else z for z in terms]
    feats = chainer(terms)
    feat_mat = frame_matrix(feats, data)
    term_map = [[feats.index(z) for z in t] for t in terms]

    # ordinally encode fixed effects
    enc_ord = OrdinalEncoder(categories='auto')
    feat_ord = enc_ord.fit_transform(feat_mat).astype(np.int)
    feat_names = [z.astype(str) for z in enc_ord.categories_]
    feat_sizes = [len(z) for z in enc_ord.categories_]

    # generate cross-matrices and cross-names
    form_vals = []
    form_names = []
    for term_idx in term_map:
        # generate cross matrices
        term_sizes = [feat_sizes[i] for i in term_idx]
        term_strides = strides(term_sizes)
        cross_vals = feat_ord[:,term_idx].dot(term_strides)
        form_vals.append(cross_vals)

        # generate cross names
        term_names = [feat_names[i] for i in term_idx]
        cross_names = [x for x in product(*term_names)]
        form_names.append(cross_names)

    # one hot encode all (cross)-terms
    hot = OneHotEncoder(categories='auto', drop=drop)
    final_mat = vstack(form_vals).T
    final_spmat = hot.fit_transform(final_mat)

    # find all cross-term names
    if hot.drop_idx_ is None:
        seen_cats = hot.categories_
    else:
        seen_cats = [np.delete(c, i) for c, i in zip(hot.categories_, hot.drop_idx_)]
    seen_names = [[n[i] for i in c] for c, n in zip(seen_cats, form_names)]
    final_names = chainer([swizzle(t, i) for i in n] for t, n in zip(terms, seen_names))

    return final_spmat, final_names

##
## absorption
##

def category_indices(cats):
    cats1 = cats.view([('', np.float)]*cats.shape[1]).squeeze()
    c_val, c_idx = np.unique(cats1, return_inverse=True)
    return c_idx

# returns forward and backward mapping
def category_maps(cats):
    cats = cats.squeeze()
    if cats.ndim == 1:
        vals = pd.Categorical(cats)
    else:
        vals = pd.Categorical(zip(*cats.T))
    group = vals._reverse_indexer()
    return vals.codes, list(group.values())

def group_sums(x, group):
    return np.array([np.bincount(group, weights=x[:, j]) for j in range(x.shape[1])])

def absorb_categorical(y, x, abs):
    N, K = x.shape
    _, A = abs.shape

    # store original means
    avg_y0 = np.mean(y)
    avg_x0 = np.mean(x, axis=0)

    # do this iteratively to reduce data loss
    for j in range(A):
        # create class groups
        codes, groups = category_maps(abs[:, j])

        # perform differencing on y
        avg_y = np.array([np.mean(y[i]) for i in groups])
        y -= avg_y[codes]

        # perform differencing on x
        avg_x = np.vstack([np.mean(x[i, :], axis=0) for i in groups])
        x -= avg_x[codes, :]

    # recenter means
    y += avg_y0
    x += avg_x0[None, :]

    return y, x

##
## design interface
##

def design_matrix(x=[], fe=[], data=None, intercept=True, drop='first', output=None):
    # construct individual matrices
    x_mat, x_names = frame_matrix(x, data), x
    c_mat, c_names = sparse_categorical(fe, data, drop=drop)

    # get data length
    N = len(data)

    # optionally add intercept
    if intercept:
        inter = np.ones((N, 1))
        x_mat = np.hstack([inter, x_mat]) if x_mat is not None else inter
        x_names = ['one'] + x_names

    # merge dense and sparse
    names = x_names + c_names
    mat = hstack([x_mat, c_mat])

    # just return if null
    if mat is None:
        return mat, names

    # handle conversions
    if output == 'sparse' and not sp.issparse(mat):
        mat = sp.csr_matrix(mat)
    if output == 'dense' and sp.issparse(mat):
        mat = mat.toarray()

    # return results
    return mat, names

def design_matrices(y, x=[], fe=[], formula=None, data=None, intercept=True, drop='first', output=None):
    if formula is not None:
        y, x, fe, intercept = parse_formula(formula)
    y_vec = frame_eval(y, data).copy()
    x_mat, x_names = design_matrix(x=x, fe=fe, data=data, intercept=intercept, drop=drop, output=output)
    return y_vec, x_mat, x_names
