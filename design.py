##
## design matrices
##

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from itertools import product

from .tools import chainer, vstack, strides, swizzle

def frame_eval(exp, data, engine='pandas'):
    if engine == 'pandas':
        return data.eval(exp).values
    elif engine == 'python':
        return eval(exp, globals(), data).values

def frame_matrix(x, data, N=None):
    return vstack([frame_eval(z, data) for z in x], N).T

def sparse_categorical(terms, data, N=None, drop='first'):
    if len(terms) == 0:
        return sp.csr_matrix((N, 0)), []

    # generate map between terms and features
    terms = [(z,) if type(z) is str else z for z in terms]
    feats = chainer(terms)
    feat_mat = frame_matrix(feats, data, N=N)
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

def design_matrix(x=[], fe=[], data=None, intercept=True, drop='first', separate=False, N=None):
    # construct individual matrices
    if len(x) > 0:
        x_mat, x_names = frame_matrix(x, data, N=N), x.copy()
    else:
        x_mat, x_names = None, []
    if len(fe) > 0:
        fe_mat, fe_names = sparse_categorical(fe, data, drop=drop, N=N)
    else:
        fe_mat, fe_names = None, []

    # try to infer N
    if N is None:
        if x_mat is not None:
            N, _ = x_mat.shape
        elif fe_mat is not None:
            N, _ = fe_mat.shape

    # we should know N by now
    if N is None:
        raise(Exception('Must specify N if no data'))

    # optionally add intercept
    if intercept:
        inter = np.ones((N, 1))
        x_mat = np.hstack([inter, x_mat]) if x_mat is not None else inter
        x_names = ['intercept'] + x_names

    # if sparse/dense separate we're done
    if separate:
        return x_mat, fe_mat, x_names, fe_names
    else:
        if x_mat is not None and fe_mat is not None:
            mat = sp.hstack([x_mat, fe_mat], format='csr')
        elif x_mat is not None and fe_mat is None:
            mat = x_mat
        elif x_mat is None and fe_spmat is not None:
            mat = fe_mat
        else:
            mat = np.empty((N, 0))
        names = x_names + fe_names
        return mat, names

def design_matrices(y, x=[], fe=[], data=None, intercept=True, drop='first', separate=False):
    y_vec = frame_eval(y, data)
    N = len(y_vec)
    x_ret = design_matrix(x, fe, data, N=N, intercept=intercept, drop=drop, separate=separate)
    return (y_vec,) + x_ret
