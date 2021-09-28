##
## formula operations
##

import re
import numpy as np
import pandas as pd

from itertools import product
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from .tools import categorize, hstack, chainer, strides

##
## tools
##

def is_categorical(ft):
    if isinstance(ft, Factor):
        return isinstance(ft, Categ)
    elif isinstance(ft, Term):
        return any([is_categorical(t) for t in ft.facts])

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

class Factor:
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return self.expr

    def __mul__(self, other):
        if isinstance(other, Factor):
            return Term((self, other))
        elif isinstance(other, Term):
            return Term((self,)+other.facts)

    def __add__(self, other):
        if isinstance(other, Factor):
            term1 = Term((self,))
            term2 = Term((other,))
            return Formula((term1, term2))
        elif isinstance(other, Term):
            term = Term((self,))
            return Formula((term, other))
        elif isinstance(other, Formula):
            term = Term((self,))
            return Formula((term,)+other.terms)

    def name(self):
        return self.expr

    def eval(self, data):
        vals = data.eval(self.expr, engine='python')
        if type(vals) is pd.Series:
            return vals.values
        elif type(vals) is np.ndarray:
            return vals
        else:
            return np.full(len(data), vals)

class Term:
    def __init__(self, facts=()):
        self.facts = tuple(facts)

    def __repr__(self):
        return self.name()

    def __iter__(self):
        return iter(self.facts)

    def __len__(self):
        return len(self.facts)

    def __mul__(self, other):
        if isinstance(other, Factor):
            return Term(self.facts+(other,))
        elif isinstance(other, Term):
            return Term(self.facts+other.facts)

    def __add__(self, other):
        if isinstance(other, Factor):
            term = Term((other,))
            return Formula((self, term))
        elif isinstance(other, Term):
            return Formula((self, other))
        elif isinstance(other, Formula):
            return Formula((self,)+other.terms)

    def name(self):
        if len(self.facts) == 0:
            return '1'
        else:
            return '*'.join([f.name() for f in self.facts])

    def raw(self, data):
        return np.vstack([f.eval(data) for f in self.facts]).T

    def eval(self, data, method='sparse', drop='first'):
        # zero length is identity
        if len(self.facts) == 0:
            return np.ones((len(data), 1)), '1'

        # separate pure real and categorical
        categ, reals = map(Term, categorize(is_categorical, self.facts))

        # handle categorical
        if len(categ) > 0:
            categ_mat = categ.raw(data)
            categ_nam = [c.expr for c in categ]
            categ_vals, categ_label = encode_categorical(
                categ_mat, categ_nam, method=method, drop=drop
            )

        # handle reals
        if len(reals) > 0:
            reals_mat = reals.raw(data)
            reals_vals = reals_mat.prod(axis=1).reshape(-1, 1)
            reals_label = reals.name()

        # combine results
        if len(categ) == 0:
            return reals_vals, [reals_label]
        elif len(reals) == 0:
            return categ_vals, categ_label
        else:
            term_vals = categ_vals.multiply(reals_vals)
            term_label = [f'{l}:{reals_label}' for l in categ_label]
            return term_vals, term_label

class Formula:
    def __init__(self, terms=()):
        self.terms = tuple(terms)

    def __repr__(self):
        return ' + '.join(str(t) for t in self.terms)

    def __iter__(self):
        return iter(self.terms)

    def __len__(self):
        return len(self.terms)

    def __add__(self, other):
        if isinstance(other, Factor):
            term = Term((other,))
            return Formula(self.terms+(term,))
        elif isinstance(other, Term):
            return Formula(self.terms+(other,))
        elif isinstance(other, Formula):
            return Formula(self.terms+other.terms)

    def eval(self, data, method='sparse', drop='first'):
        # split by all real or not
        categ, reals = categorize(is_categorical, self.terms)

        # handle categories
        if len(categ) > 0:
            categ_vals, categ_label = zip(*[
                t.eval(data, method=method, drop=drop) for t in categ
            ])
            categ_vals = hstack(categ_vals)
        else:
            categ_vals, categ_label = None, []

        # combine labels
        if method == 'sparse':
            categ_label = chainer(categ_label)
        elif method == 'ordinal':
            categ_label = {t.name(): ls for t, ls in zip(categ, categ_label)}

        # handle reals
        if len(reals) > 0:
            reals_vals, reals_label = zip(*[t.eval(data) for t in reals])
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
    pass

class Categ(Factor):
    def __repr__(self):
        return f'C({self.expr})'

# shortcuts
I = Term()
R = Real
C = Categ

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
    return Term([parse_factor(f) for f in term.factors])

# this can only handle treatment coding, but that's required for sparsity
def parse_formula(form):
    from patsy.desc import ModelDesc

    # use patsy for formula parse
    desc = ModelDesc.from_formula(form)
    lhs, rhs = desc.lhs_termlist, desc.rhs_termlist

    # convert to string lists
    x_terms = Formula([parse_term(t) for t in rhs])
    if len(lhs) > 0:
        y_terms = parse_factor(lhs[0].factors[0])
        return y_terms, x_terms
    else:
        return x_terms

def parse_item(i):
    if isinstance(i, Factor):
        return i
    else:
        return Factor(i)

def parse_tuple(t):
    if isinstance(t, Term):
        return t
    else:
        if type(t) not in (tuple, list):
            t = t,
        return Term([parse_item(i) for i in t])

def parse_list(l):
    if isinstance(l, Formula):
        return l
    else:
        if type(l) not in (tuple, list):
            l = l,
        return Formula([parse_tuple(t) for t in l])

##
## design interface
##

def design_matrices(
    y=None, x=None, formula=None, data=None, method='sparse', drop='first'
):
    if formula is not None:
        y, x = parse_formula(formula)
    else:
        y, x = parse_item(y), parse_list(x)

    y_vec, y_name = y.eval(data), y.expr
    x_mat, x_names, c_mat, c_labels = x.eval(data, method=method, drop=drop)

    return y_vec, y_name, x_mat, x_names, c_mat, c_labels
