##
## design matrices
##

import numpy as np
import pandas as pd
import scipy.sparse as sp
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
## design
##

def frame_eval(exp, data, engine='pandas'):
    if engine == 'pandas':
        return data.eval(exp).values
    elif engine == 'python':
        return eval(exp, globals(), data).values

def frame_matrix(terms, data):
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

# absorb categorical variables
def absorb_categorical(y, x, categ):
    N, K = categ.shape

    # iteratively difference out
    for c in range(K):
        vals = pd.Categorical(categ[:, c])
        group = vals._reverse_indexer()
        first = {k: v[0] for k, v in group.items()}
        idx = np.array([first[x] for x in vals])
        y -= y[idx]
        x -= x[idx, :]

    return y, x

def design_matrix(x=[], fe=[], data=None, intercept=True, drop='first', output=None):
    # construct individual matrices
    x_mat, x_names = frame_matrix(x, data), x
    fe_mat, fe_names = sparse_categorical(fe, data, drop=drop)

    # get data length
    N = len(data)

    # optionally add intercept
    if intercept:
        inter = np.ones((N, 1))
        x_mat = np.hstack([inter, x_mat]) if x_mat is not None else inter
        x_names = ['intercept'] + x_names

    # merge dense and sparse
    names = x_names + fe_names
    mat = hstack([x_mat, fe_mat])

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

def design_matrices(y, x=[], fe=[], ab=[], data=None, intercept=True, drop='first', output=None):
    # can't use intercept with absorbtion
    if len(ab) > 0:
        intercept = False

    # base matrices
    y_vec = frame_eval(y, data)
    x_mat, x_names = design_matrix(x, fe, data, intercept=intercept, drop=drop, output=output)

    # absorb desired categoricals
    if len(ab) > 0:
        x_abs = frame_matrix(ab, data)
        y_vec, x_mat = absorb_categorical(y_vec, x_mat, x_abs)

    return y_vec, x_mat, x_names
