##
## formula operations
##

import re
import numpy as np
import pandas as pd

from itertools import product
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from .tools import (
    categorize, hstack, chainer, strides, decorator, func_name, func_disp
)

##
## tools
##

def is_categorical(ft):
    if isinstance(ft, Factor):
        return isinstance(ft, Categ)
    elif isinstance(ft, Term):
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

# ordinally encode interactions terms (tuple-like things)
def category_indices(vals, return_labels=False):
    if vals.ndim == 1:
        vals = vals[:, None]

    # convert to packed integers
    ord_enc = OrdinalEncoder(categories='auto', dtype=int)
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
def encode_categorical(vals, names, method='sparse', drop='first'):
    # reindex categoricals jointly
    categ_vals, categ_labels = category_indices(vals, return_labels=True)
    categ_vals = categ_vals.reshape(-1, 1)
    categ_labels = [swizzle(names, l) for l in categ_labels]

    # encode indices with chosen method
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

    # get used lables
    cats_used = [categ_labels[i] for i in cats_all]

    return cats_enc, cats_used

##
## formula structure
##

class AccessorType(type):
    def __getattr__(cls, expr):
        return cls(expr)

class Factor(metaclass=AccessorType):
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
        return str(self) == str(other)

    def __repr__(self):
        return self._name

    def __add__(self, other):
        if isinstance(other, (Factor, Term)):
            return Formula(self, other)
        elif isinstance(other, Formula):
            return Formula(self, *other)

    def __mul__(self, other):
        if isinstance(other, Factor):
            return Term(self, other)
        elif isinstance(other, Term):
            return Term(self, *other)
        elif isinstance(other, Formula):
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

class Term:
    def __init__(self, *facts):
        self._facts = facts

    def __hash__(self):
        return hash(tuple(set(self)))

    def __eq__(self, other):
        return set(self) == set(other)

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
        if isinstance(other, (Factor, Term)):
            return Formula(self, other)
        elif isinstance(other, Formula):
            return Formula(self, *other)

    def __mul__(self, other):
        if isinstance(other, Factor):
            return Term(*self, other)
        elif isinstance(other, Term):
            return Term(*self, *other)
        elif isinstance(other, Formula):
            return Formula(*[Term(*self, *t) for t in other])

    def name(self):
        return '*'.join([f.name() for f in self])

    def raw(self, data, extern=None):
        return np.vstack([f.eval(data, extern=extern) for f in self]).T

    def enc(self, data):
        return category_indices(self.raw(data))

    def eval(self, data, method='sparse', drop='first', extern=None):
        # zero length is identity
        if len(self) == 0:
            return np.ones((len(data), 1)), 'I'

        # separate pure real and categorical
        categ, reals = categorize(is_categorical, self)
        categ, reals = Term(*categ), Term(*reals)

        # handle categorical
        if len(categ) > 0:
            categ_mat = categ.raw(data, extern=extern)
            categ_nam = [c.name() for c in categ]
            categ_vals, categ_label = encode_categorical(
                categ_mat, categ_nam, method=method, drop=drop
            )

        # handle reals
        if len(reals) > 0:
            reals_mat = reals.raw(data, extern=extern)
            reals_vals = reals_mat.prod(axis=1).reshape(-1, 1)
            reals_label = reals.name()

        # combine results
        if len(categ) == 0:
            return reals_vals, [reals_label]
        elif len(reals) == 0:
            return categ_vals, categ_label
        else:
            term_vals = categ_vals.multiply(reals_vals)
            term_label = [f'({l})*{reals_label}' for l in categ_label]
            return term_vals, term_label

class Formula:
    def __init__(self, *terms):
        self._terms = tuple(dict.fromkeys(
            t if isinstance(t, Term) else Term(t) for t in terms
        )) # order preserving unique

    def __repr__(self):
        return ' + '.join(str(t) for t in self)

    def __iter__(self):
        return iter(self._terms)

    def __len__(self):
        return len(self._terms)

    def __add__(self, other):
        if isinstance(other, (Factor, Term)):
            return Formula(*self, other)
        elif isinstance(other, Formula):
            return Formula(*self, *other)

    def __sub__(self, other):
        if isinstance(other, Factor):
            other = Term(other)
        if isinstance(other, Term):
            return Formula(*[
                t for t in self if t != other
            ])

    def __mul__(self, other):
        if isinstance(other, Factor):
            return Formula(*[Term(*t, other) for t in self])
        elif isinstance(other, Term):
            return Formula(*[Term(*t, *other) for t in self])
        elif isinstance(other, Formula):
            return Formula(*chainer([
                [Term(*t1, *t2) for t1 in self] for t2 in other
            ]))

    def enc(self, data):
        return np.vstack([t.enc(data) for t in self]).T

    def eval(self, data, method='sparse', drop='first', extern=None):
        # split by all real or not
        categ, reals = categorize(is_categorical, self)

        # handle categories
        if len(categ) > 0:
            categ_vals, categ_label = zip(*[
                t.eval(data, method=method, drop=drop, extern=extern)
                for t in categ
            ])
            categ_vals = hstack(categ_vals)
        else:
            categ_vals, categ_label = None, []

        # combine labels
        categ_label = {t: ls for t, ls in zip(categ, categ_label)}

        # handle reals
        if len(reals) > 0:
            reals_vals, reals_label = zip(*[
                t.eval(data, extern=extern) for t in reals
            ])
            reals_vals = hstack(reals_vals)
            reals_label = chainer(reals_label)
        else:
            reals_vals, reals_label = None, []

        # return separately
        return reals_vals, reals_label, categ_vals, categ_label

##
## column types
##

class Real(Factor):
    def __repr__(self):
        return f'R({self.name()})'

class Categ(Factor):
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

    # convert to string lists
    x_terms = Formula(*[parse_term(t) for t in rhs])
    if len(lhs) > 0:
        y_terms = parse_factor(lhs[0].factors[0])
        return y_terms, x_terms
    else:
        return x_terms

def parse_item(i, convert=Real):
    if isinstance(i, Factor):
        return i
    else:
        return convert(i)

def parse_tuple(t, convert=Real):
    if isinstance(t, Term):
        return t
    else:
        if type(t) not in (tuple, list):
            t = t,
        return Term(*[
            parse_item(i, convert=convert) for i in t
        ])

def parse_list(l, convert=Real):
    if isinstance(l, Formula):
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
        y, x = parse_item(y), parse_list(x)
    return y, x

def design_matrices(
    y=None, x=None, formula=None, data=None, method='sparse', drop='first',
    extern=None
):
    y, x = ensure_formula(x=x, y=y, formula=formula)
    y_vec, y_name = y.eval(data, extern=extern), y.name()
    x_mat, x_names, c_mat, c_labels = x.eval(
        data, method=method, drop=drop, extern=extern
    )
    return y_vec, y_name, x_mat, x_names, c_mat, c_labels
