##
## formula operations
##

import re
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .meta import MetaFactor, MetaTerm, MetaFormula, MetaReal, MetaCateg
from .tools import (
    categorize, hstack, chainer, decorator, func_disp, valid_rows, split_size,
    atleast_2d, fillna, all_valid, splice, factorize_2d, onehot_encode
)

##
## tools
##

def is_categorical(ft):
    if isinstance(ft, MetaFactor):
        return isinstance(ft, MetaCateg)
    elif isinstance(ft, MetaTerm):
        return any(is_categorical(t) for t in ft)

def ensure_tuple(t):
    if type(t) is tuple:
        return t
    elif type(t) is list:
        return tuple(t)
    else:
        return t,

def robust_eval(data, expr, extern=None):
    # short circuit
    if expr is None:
        return None

    # extract values
    if type(expr) is str:
        vals = data.eval(expr, engine='python', local_dict=extern)
    elif callable(expr):
        vals = expr(data)
    else:
        vals = expr

    # ensure array
    if type(vals) is pd.Series:
        return vals.values
    elif type(vals) is np.ndarray:
        return vals
    else:
        return np.full(len(data), vals)

##
## categoricals
##

# make labels
def swizzle(ks, vs):
    return ','.join([f'{k}={v}' for k, v in zip(ks, vs)])

# ordinally encode interaction terms (tuple-like things)
# null data is returned with a -1 index if not dropna
def category_indices(vals, dropna=False, return_labels=False):
    # also accept single vectors
    vals = atleast_2d(vals)

    # track valid rows
    valid = valid_rows(vals)
    vals1 = vals[valid]

    # find unique rows
    uni_indx, uni_vals = factorize_2d(vals1)

    # patch in valid data
    if dropna or valid.all():
        uni_ind1 = uni_indx
    else:
        uni_ind1 = splice(valid, uni_indx, -1)

    # return requested
    if return_labels:
        return uni_ind1, uni_vals, valid
    else:
        return uni_ind1, valid

# encode categories as one-hot matrix
def encode_categorical(vals, names, method='sparse', drop=True):
    # reindex categoricals jointly
    cats_val, cats_lab, valid = category_indices(vals, return_labels=True)
    cats_lab = [swizzle(names, l) for l in cats_lab]

    # if ordinal no labels are dropped
    if method == 'ordinal':
        cats_enc, cats_use = cats_val.reshape(-1, 1), cats_lab
    elif method == 'sparse':
        cats_enc, cats_all = onehot_encode(cats_val, drop=drop)
        cats_use = [cats_lab[i] for i in cats_all]

    return cats_enc, cats_use, valid

# subset data allowing for missing chunks
def drop_invalid(valid, *mats, warn=True):
    V, N = np.sum(valid), len(valid)
    if V == 0:
        raise Exception('all rows contain null data')
    elif V < N:
        if warn:
            print(f'dropping {N-V}/{N} null data points')
        mats = [
            m[valid] if m is not None else None for m in mats
        ]
    return *mats,

# remove unused categories (sparse `mat` requires `labels`)
def prune_categories(mat, labels=None, method='sparse', warn=True):
    # get positive count cats
    if method == 'sparse':
        vcats = np.ravel((mat!=0).sum(axis=0)) > 0
        Kt = [len(ls) for ls in labels.values()]
        tvalid = split_size(vcats, Kt)
        P, K = np.sum(vcats), len(vcats)
    elif method == 'ordinal':
        vcats = [np.bincount(cm) > 0 for cm in mat.T]
        P = sum([np.sum(vc) for vc in vcats])
        K = sum([len(vc) for vc in vcats])

    # prune if needed
    if P < K:
        if warn:
            print(f'pruning {K-P}/{K} unused categories')

        # modify data matrix
        if method == 'sparse':
            mat = mat[:, vcats]
            labels = {
                t: [l for l, v in zip(ls, vs) if v]
                for (t, ls), vs in zip(labels.items(), tvalid)
            }
        elif method == 'ordinal':
            mat = np.vstack([category_indices(cm) for cm in mat.T]).T

    return mat, labels

##
## formula structure
##

class AccessorType(type):
    def __getattr__(cls, expr):
        return cls(expr)

class Factor(MetaFactor, metaclass=AccessorType):
    def __init__(self, expr, name=None):
        if type(expr) is str:
            self._expr = expr
            self._name = expr if name is None else name
        elif callable(expr):
            self._expr = expr
            self._name = name
        elif type(expr) is pd.Series:
            self._expr = expr.values
            self._name = expr.name
        else:
            self._expr = np.array(expr)
            self._name = name

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, MetaFactor):
            return str(self) == str(other)
        else:
            return False

    def __repr__(self):
        return self._name

    def __add__(self, other):
        if isinstance(other, (MetaFactor, MetaTerm)):
            return Formula(self, other)
        elif isinstance(other, MetaFormula):
            return Formula(self, *other)

    def __sub__(self, other):
        return Formula(self) - other

    def __mul__(self, other):
        if isinstance(other, MetaFactor):
            return Term(self, other)
        elif isinstance(other, MetaTerm):
            return Term(self, *other)
        elif isinstance(other, MetaFormula):
            return Formula(*[Term(self, *t) for t in other])

    def __call__(self, *args, **kwargs):
        cls = type(self)
        return cls(self._expr, *args, **kwargs)

    def to_term(self):
        return Term(self)

    def name(self):
        return self._name

    def eval(self, data=None, extern=None):
        return robust_eval(data, self._expr, extern=extern)

class Term(MetaTerm):
    def __init__(self, *facts):
        self._facts = facts

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, MetaTerm):
            return set(self) == set(other)
        elif isinstance(other, MetaFactor):
            return set(self) == {other}
        else:
            return False

    def __repr__(self):
        if len(self) == 0:
            return 'I'
        else:
            return '*'.join([str(f) for f in self])

    def __iter__(self):
        return iter(self._facts)

    def __len__(self):
        return len(self._facts)

    def __add__(self, other):
        if isinstance(other, (MetaFactor, MetaTerm)):
            return Formula(self, other)
        elif isinstance(other, MetaFormula):
            return Formula(self, *other)

    def __sub__(self, other):
        return Formula(self) - other

    def __mul__(self, other):
        if isinstance(other, MetaFactor):
            return Term(*self, other)
        elif isinstance(other, MetaTerm):
            return Term(*self, *other)
        elif isinstance(other, MetaFormula):
            return Formula(*[Term(*self, *t) for t in other])

    def name(self):
        return '*'.join([f.name() for f in self])

    def raw(self, data, extern=None):
        return np.vstack([f.eval(data, extern=extern) for f in self]).T

    def eval(self, data, method='sparse', drop=True, extern=None):
        # zero length is identity
        if len(self) == 0:
            N = len(data)
            return np.ones((N, 1)), np.ones(N, dtype=bool), ['I']

        # separate pure real and categorical
        categ, reals = categorize(is_categorical, self)
        categ, reals = Term(*categ), Term(*reals)

        # handle categorical
        if len(categ) > 0:
            categ_mat = categ.raw(data, extern=extern)
            categ_nam = [c.name() for c in categ]
            categ_value, categ_label, categ_valid = encode_categorical(
                categ_mat, categ_nam, method=method, drop=drop
            )

        # handle reals
        if len(reals) > 0:
            reals_mat = reals.raw(data, extern=extern)
            reals_value = reals_mat.prod(axis=1).reshape(-1, 1)
            reals_label = reals.name()
            reals_valid = valid_rows(reals_value)

        # combine results
        if len(categ) == 0:
            return reals_value, reals_valid, [reals_label]
        elif len(reals) == 0:
            return categ_value, categ_valid, categ_label
        else:
            # filling nulls with 0 keeps sparse the same
            term_value = categ_value.multiply(fillna(reals_value, v=0))
            term_label = [f'({l})*{reals_label}' for l in categ_label]
            term_valid = categ_valid & reals_valid
            return term_value, term_valid, term_label

class Formula(MetaFormula):
    def __init__(self, *terms):
        self._terms = tuple(dict.fromkeys(
            t if isinstance(t, MetaTerm) else Term(t) for t in terms
        )) # order preserving unique

    def __repr__(self):
        return ' + '.join(str(t) for t in self)

    def __iter__(self):
        return iter(self._terms)

    def __len__(self):
        return len(self._terms)

    def __add__(self, other):
        if isinstance(other, (MetaFactor, MetaTerm)):
            return Formula(*self, other)
        elif isinstance(other, MetaFormula):
            return Formula(*self, *other)

    def __sub__(self, other):
        if isinstance(other, MetaFactor):
            other = Term(other)
        if isinstance(other, MetaTerm):
            return Formula(*[
                t for t in self if t != other
            ])
        if isinstance(other, MetaFormula):
            return Formula(*[
                t for t in self if t not in other
            ])

    def __mul__(self, other):
        if isinstance(other, MetaFactor):
            return Formula(*[Term(*t, other) for t in self])
        elif isinstance(other, MetaTerm):
            return Formula(*[Term(*t, *other) for t in self])
        elif isinstance(other, MetaFormula):
            return Formula(*chainer([
                [Term(*t1, *t2) for t1 in self] for t2 in other
            ]))

    def raw(self, data, extern=None):
        return [t.raw(data, extern=extern) for t in self]

    def eval(self, data, method='sparse', drop=True, extern=None):
        # split by all real or not
        categ, reals = categorize(is_categorical, self)

        # handle categories
        if len(categ) > 0:
            categ_value, categ_valid, categ_label = zip(*[
                t.eval(data, method=method, drop=drop, extern=extern)
                for t in categ
            ])
            categ_value = hstack(categ_value)
            categ_valid = np.vstack(categ_valid).all(axis=0)
        else:
            categ_value, categ_valid, categ_label = None, None, []

        # combine labels
        categ_label = {t: ls for t, ls in zip(categ, categ_label)}

        # handle reals
        if len(reals) > 0:
            reals_value, reals_valid, reals_label = zip(*[
                t.eval(data, extern=extern) for t in reals
            ])
            reals_value = hstack(reals_value)
            reals_label = chainer(reals_label)
            reals_valid = np.vstack(reals_valid).all(axis=0)
        else:
            reals_value, reals_valid, reals_label = None, None, []

        # return separately
        return (
            reals_value, reals_valid, reals_label,
            categ_value, categ_valid, categ_label
        )

##
## column types
##

class Real(MetaReal, Factor):
    def __repr__(self):
        return f'R({self.name()})'

class Categ(MetaCateg, Factor):
    def __repr__(self):
        return f'C({self.name()})'

# custom columns — class interface
# eval (mandatory): an ndarray of the values
# name (recommended): what gets displayed in the regression table
# __repr__ (optional): what gets displayed on print [default to C/R(name)]

class Demean(Real):
    def __init__(self, expr, cond=None, name=None):
        args = '' if cond is None else f'|{cond}'
        name = expr if name is None else name
        super().__init__(expr, name=f'{name}-μ{args}')
        self._cond = cond

    def eval(self, data, extern=None):
        vals = super().eval(data, extern=extern)
        if self._cond is None:
            means = np.mean(vals)
        else:
            cond = robust_eval(data, self._cond, extern=extern)
            datf = pd.DataFrame({'vals': vals, 'cond': cond})
            cmean = datf.groupby('cond')['vals'].mean().rename('mean')
            datf = datf.join(cmean, on='cond')
            means = datf['mean'].values
        return vals - means

class Binned(Categ):
    def __init__(self, expr, bins=10, labels=False, name=None):
        nb = bins if type(bins) is int else len(bins)
        name = expr if name is None else name
        super().__init__(expr, name=f'{name}:bin{nb}')
        self._bins = bins
        self._labels = None if labels else False

    def eval(self, data, extern=None):
        vals = super().eval(data, extern=extern)
        bins = pd.cut(vals, self._bins, labels=self._labels)
        return bins

# custom columns — functional interface

class Custom:
    def __init__(
        self, func, name=None, categ=False, base=Real, eval_args=0, frame=False
    ):
        self._base = Categ if categ else base
        self._func = func
        self._name = name if callable(name) else func_disp(func, name=name)
        self._eval = ensure_tuple(eval_args)
        self._frame = frame

    def __getattr__(self, key):
        if key.startswith('_'):
            return super().__getattr__(key)
        else:
            return self(key)

    def __call__(self, *args, **kwargs):
        if self._frame:
            evaler = lambda data: self._func(data, *args, **kwargs)
        else:
            def evaler(data):
                args1 = [
                    robust_eval(data, e) if i in self._eval else e
                    for i, e in enumerate(args)
                ]
                return self._func(*args1, **kwargs)
        name = self._name(*args, **kwargs)
        return self._base(evaler, name=name)

@decorator
def factor(func, *args, **kwargs):
    return Custom(func, *args, **kwargs)

# shortcuts
I = Term()
R = Real
C = Categ
D = Demean
B = Binned

##
## conversion
##

# lookup table
FTYPES = {
    'C': Categ,
    'I': Real,
}

def parse_factor(fact):
    ret = re.match(r'(C|I)\(([^\)]+)\)', fact.code)
    if ret is not None:
        pre, name = ret.groups()
        return FTYPES[pre](name)
    else:
        return Real(fact.code)

def parse_term(term):
    return Term(*[parse_factor(f) for f in term.factors])

# this can only handle treatment coding, but that's required for sparsity
def parse_formula(form):
    try:
        from patsy.desc import ModelDesc
    except:
        print('Please install patsy for formula parsing')
        return

    # use patsy for formula parse
    desc = ModelDesc.from_formula(form)
    lhs, rhs = desc.lhs_termlist, desc.rhs_termlist

    # check for invalid y
    if len(lhs) > 1:
        raise Exception('Must have single factor y term')

    # convert to string lists
    y_terms = parse_factor(lhs[0].factors[0]) if len(lhs) > 0 else None
    x_terms = Formula(*[parse_term(t) for t in rhs])

    return y_terms, x_terms

def parse_item(i, convert=Real):
    if isinstance(i, MetaFactor):
        return i
    else:
        return convert(i)

def parse_tuple(t, convert=Real):
    if isinstance(t, MetaTerm):
        return t
    else:
        if type(t) not in (tuple, list):
            t = t,
        return Term(*[
            parse_item(i, convert=convert) for i in t
        ])

def parse_list(l, convert=Real):
    if isinstance(l, MetaFormula):
        return l
    else:
        if type(l) not in (tuple, list):
            l = l,
        return Formula(*[
            parse_tuple(t, convert=convert) for t in l
        ])

##
## design interface
##

def ensure_formula(y=None, x=None, formula=None):
    if formula is not None:
        y, x = parse_formula(formula)
    else:
        y = parse_item(y) if y is not None else None
        x = parse_list(x)
    return y, x

def design_matrix(
    y=None, x=None, formula=None, data=None, method='sparse', drop=True,
    dropna=True, prune=True, warn=True, extern=None, valid0=None, flatten=True,
    validate=False
):
    y, x = ensure_formula(x=x, formula=formula)
    if y is not None:
        raise Exception('Use design_matrices for formulas with an LHS.')

    # evaluate x variables
    x_mat, x_val, x_names, c_mat, c_val, c_labels = x.eval(
        data, method=method, drop=drop, extern=extern
    )

    # aggregate valid info for data
    valid = all_valid(valid0, x_val, c_val)

    # drop null values if requested
    if dropna:
        x_mat, c_mat = drop_invalid(
            valid, x_mat, c_mat, warn=warn
        )

    # prune empty categories if requested
    if prune and c_mat is not None:
        c_mat, c_labels = prune_categories(
            c_mat, c_labels, method=method, warn=warn
        )

    # combine real and categorical?
    if flatten:
        f_mat = hstack([x_mat, c_mat])
        f_names = x_names + chainer(c_labels.values())
        ret = f_mat, f_names
    else:
        ret = x_mat, x_names, c_mat, c_labels

    # return valid mask?
    if validate:
        return *ret, valid
    else:
        return ret

def design_matrices(
    y=None, x=None, formula=None, data=None, dropna=True, extern=None,
    valid0=None, validate=False, **kwargs
):
    # parse into pythonic formula system
    y, x = ensure_formula(x=x, y=y, formula=formula)
    if y is None:
        raise Exception('Use design_matrix for formulas without an LHS')

    # get y data
    y_vec, y_name = y.eval(data, extern=extern), y.name()
    y_val = valid_rows(y_vec)

    # get valid x data
    x_val0 = all_valid(valid0, y_val)
    *x_ret, valid = design_matrix(
        x=x, data=data, dropna=dropna, extern=extern, valid0=x_val0,
        validate=True, **kwargs
    )

    # drop invalid y
    if dropna:
        y_vec, = drop_invalid(valid, y_vec)

    # return combined data
    ret = y_vec, y_name, *x_ret
    if validate:
        return *ret, valid
    else:
        return ret
