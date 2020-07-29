import jax
import jax.lax as lax
import jax.scipy.special as spec
import jax.numpy as np
from jax.tree_util import tree_flatten, tree_leaves, tree_map, tree_multimap, tree_reduce, tree_structure
from jax.interpreters.xla import DeviceArray
import numpy as np0
import scipy.sparse as sp
import pandas as pd
from operator import and_, add

from .design import design_matrices
from .summary import param_table

##
## constants
##

# numbers
eps = 1e-7
clip_like = 20.0

# polygamma functions
@jax.custom_transforms
def trigamma(x):
    return 1/x + 1/(2*x**2) + 1/(6*x**3) - 1/(30*x**5) + 1/(42*x**7) - 1/(30*x**9) + 5/(66*x**11) - 691/(2730*x**13) + 7/(6*x**15)

@jax.custom_transforms
def digamma(x):
    return spec.digamma(x)

@jax.custom_transforms
def gammaln(x):
    return spec.gammaln(x)

jax.defjvp(digamma, lambda g, y, x: lax.mul(g, trigamma(x)))
jax.defjvp(gammaln, lambda g, y, x: lax.mul(g, digamma(x)))

##
## precompile
##

inv_fun = jax.jit(np.linalg.inv)

##
## canned models
##

def sigmoid(x):
    return 1/(1+np.exp(-x))

def log(x):
    return np.log(np.maximum(eps, x))

# link functions
links = {
    'ident': lambda x: x,
    'exp': np.exp,
    'logit': sigmoid
}

# loss functions
def binary_loss(yh, y):
    return y*log(yh) + (1-y)*log(1-yh)

def poisson_loss(yh, y):
    return y*log(yh) - yh

def negbin_loss(r, yh, y):
    return gammaln(r+y) - gammaln(r) + r*log(r) + y*log(yh) - (r+y)*log(r+yh)

def lstsq_loss(yh, y):
    return -(y-yh)**2

losses = {
    'binary': lambda p, yh, y: binary_loss(yh, y),
    'poisson': lambda p, yh, y: poisson_loss(yh, y),
    'negbin': lambda p, yh, y: negbin_loss(np.exp(p['lr']), yh, y),
    'lstsq': lambda p, yh, y: lstsq_loss(yh, y)
}

# modifiers
def zero_inflate(like0, key='lpzero'):
    def like(p, yh, y):
        pzero = sigmoid(p[key])
        clike = np.clip(like0(p, yh, y), a_max=clip_like)
        like = pzero*(y==0) + (1-pzero)*np.exp(clike)
        return log(like)
    return like

##
## batching it, pytree style
##

class DataLoader:
    def __init__(self, data, batch_size=None):
        if type(data) is pd.DataFrame:
            data = data.to_dict('series')
        self.data = tree_map(lambda x: np.array(x) if type(x) is not np.ndarray else x, data)
        shapes = [d.shape[0] for d in tree_leaves(self.data)]
        self.data_size = shapes[0] # should all be the same size
        self.batch_size = batch_size

    def __call__(self, batch_size=None):
        yield from self.iterate(batch_size)

    def __iter__(self):
        yield from self.iterate()

    def iterate(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_batches = max(1, self.data_size // batch_size)
        round_size = batch_size*num_batches
        for i in range(0, round_size, batch_size):
            yield tree_map(lambda d: d[i:i+batch_size], self.data)

##
## block matrices
##

def chunks(v, n):
    return [v[i:i+n] for i in range(0, len(v), n)]

# need to handle ambiguity in case x.ndim == 1
def atleast_2d(x, axis=0):
    x2 = np.atleast_2d(x)
    if x.ndim == 1 and axis == 0:
        x2 = x2.T
    return x2

def block_matrix(tree, dims, size):
    blocks = chunks(tree_leaves(tree), size)
    mat = np.block([[atleast_2d(x, axis=int(d>0)) for d, x in zip(dims, row)] for row in blocks])
    return mat

def block_unpack(mat, tree, sizes):
    part = np.cumsum(np.array(sizes))
    block = lambda x, axis: tree.unflatten(np.split(x, part, axis=axis)[:-1])
    tree = tree_map(lambda x: block(x, 1), block(mat, 0))
    return tree

# we need par to know the inner shape
def tree_matfun(fun, mat, par):
    # get param configuration
    par_flat, par_tree = tree_flatten(par)
    par_sizs = [np.size(p) for p in par_flat]
    par_dims = [np.ndim(p) for p in par_flat]
    K = len(par_flat)

    # invert hessian for stderrs
    tmat = block_matrix(mat, par_dims, K)
    fmat = fun(tmat)
    fout = block_unpack(fmat, par_tree, par_sizs)

    return fout

def tree_batch_reduce(batch_fun, loader, agg_fun=np.add):
    total = None
    for b, batch in enumerate(loader):
        f_batch = batch_fun(batch)
        if total is None:
            total = f_batch
        else:
            total = tree_multimap(agg_fun, total, f_batch)
    return total

def tree_outer(tree):
    return tree_map(lambda x: tree_map(lambda y: x.T @ y, tree), tree)

def popoff(d, s):
    if s in d:
        return {k: v for k, v in d.items() if k != s}, d[s]
    else:
        return d, None

def tree_hessian(model, params, loader, method='deriv', batch_size=1024):
    N = loader.data_size
    B = N // batch_size
    batch_iter = loader(batch_size)

    if method == 'deriv':
        h_fun = jax.jit(jax.hessian(model))
        hess = tree_batch_reduce(lambda b: h_fun(params, b), batch_iter)
        hess = tree_map(lambda x: x/B, hess)
    elif method == 'outer':
        tree_axis = tree_map(lambda _: 0, loader.data)
        g_fun = jax.jit(jax.vmap(jax.grad(model), (None, tree_axis), 0))
        hess = tree_batch_reduce(lambda b: tree_outer(g_fun(params, b)), batch_iter)
        hess = tree_map(lambda x: -x/N, hess)
    else:
        raise Exception(f'Unknown hessian method: {method}')

    return hess

##
## optimizers
##

def adam(vg_fun, loader, params0, epochs=3, eta=0.01, gamma=0.9, disp=None):
    # parameter info
    params = tree_map(np.array, params0)

    # track rms gradient
    grms = tree_map(np.zeros_like, params)

    # do training
    for ep in range(epochs):
        # epoch stats
        agg_loss, agg_batch = 0.0, 0

        # iterate over batches
        for b, batch in enumerate(loader):
            # compute gradients
            loss, grad = vg_fun(params, batch)

            lnan = np.isnan(loss)
            gnan = tree_reduce(and_, tree_map(lambda g: np.isnan(g).any(), grad))

            if lnan or gnan:
                print('Encountered nans!')
                return params, None

            grms = tree_multimap(lambda r, g: gamma*r + (1-gamma)*g**2, grms, grad)
            params = tree_multimap(lambda p, g, r: p + eta*g/np.sqrt(r+eps), params, grad, grms)

            # compute statistics
            agg_loss += loss
            agg_batch += 1

        # display stats
        avg_loss = agg_loss/agg_batch

        # display output
        if disp is not None:
            disp(ep, avg_loss, params)

    return params

##
## estimation
##

def tree_outer_flat(tree):
    tree1, vec = popoff(tree, 'hdfe')
    mat = np.hstack(tree_leaves(tree1))
    A = mat.T @ mat
    B = mat.T @ vec
    C = vec.T @ mat
    d = np.sum(vec*vec, axis=0)
    return A, B, C, d

# maximum likelihood using jax - this expects a mean log likelihood
# the assumes the data is batchable, which usually means panel-like
# a toplevel hdfe variable is treated special-like
def maxlike(model=None, params=None, data=None, vg_fun=None, hessian='outer', stderr=False, optim=adam, batch_size=8192, batch_stderr=8192, backend='gpu', **kwargs):
    if vg_fun is None:
        vg_fun = jax.jit(jax.value_and_grad(model))

    # set up batching
    loader = DataLoader(data, batch_size)
    N = loader.data_size

    # maximize likelihood
    params1 = optim(vg_fun, loader, params, **kwargs)

    if not stderr:
        return params1, None

    if 'hdfe' in params:
        # get vectorized gradient
        tree_axis = tree_map(lambda _: 0, data)
        vg_fun = jax.jit(jax.vmap(jax.grad(model), (None, tree_axis), 0), backend=backend)

        # compute hessian blocks
        hload = loader(batch_stderr)
        A, B, C, d = tree_batch_reduce(lambda b: tree_outer_flat(vg_fun(params1, b)), hload)
        d1 = 1/d
        d1l, d1r = d1[:, None], d1[None, :]

        # use block inverse formula
        A1 = inv_fun(A - (B*d1r) @ C)
        psig = A1
        hsig = d1 + np.sum((d1l*C)*(A1 @ (B*d1r)).T, axis=1)

        # unpack into tree
        par0, _ = popoff(params1, 'hdfe')
        par0_tree = tree_structure(par0)
        par0_sizs = [np.size(p) for p in tree_leaves(par0)]
        sigma = block_unpack(psig, par0_tree, par0_sizs)
        sigma['hdfe'] = hsig
    else:
        # get model hessian
        hess = tree_hessian(model, params, loader, method=hessian, batch_size=batch_stderr)
        fish = tree_matfun(inv_fun, hess, params)
        sigma = tree_map(lambda x: -x/N, fish)

    return params1, sigma

# make a glm model and compile
def glm_model(link, loss, hdfe=None):
    if type(link) is str:
        link = links[link]
    if type(loss) is str:
        loss = losses[loss]

    # evaluator
    def model(par, dat):
        ydat, xdat, cdat = dat['ydat'], dat['xdat'], dat['cdat']
        real, categ = par['real'], par['categ']
        if hdfe is not None:
            categ[hdfe] = par.pop('hdfe')
        linear = xdat @ real
        for i, c in enumerate(categ):
            cidx = cdat.T[i] # needed for vmap to work
            linear += categ[c][cidx]
        pred = link(linear)
        like = loss(par, pred, ydat)
        return np.mean(like)

    return model

# default glm specification
def glm(y, x=[], fe=[], hdfe=None, data=None, extra={}, model=None, link=None, loss=None, intercept=True, stderr=True, **kwargs):
    if hdfe is not None:
        if type(hdfe) is list:
            raise Exception('Can\'t handle more than one HD FE')
        fe += [hdfe]

    # construct design matrices
    y_vec, x_mat, x_names, c_mat, c_names = design_matrices(y, x=x, fe=fe, data=data, intercept=intercept, method='ordinal')

    # get data shape
    N = len(y_vec)
    Kx = len(x_names)
    Kc = len(c_names)
    Kl = [len(c) for c in c_names.values()]

    # compile model if needed
    if model is None:
        model = glm_model(link, loss, hdfe=hdfe)

    # displayer
    def disp(e, l, p):
        real, categ = p['real'], p['categ']
        if hdfe is not None:
            categ = categ.copy()
            categ[hdfe] = p['hdfe']
        mcats = np.array([np.mean(c) for c in categ.values()])
        print(f'[{e:3d}] {l:.4f}: {np.mean(real):.4f} {np.mean(mcats):.4f}')

    # organize data
    data = {'ydat': y_vec, 'xdat': x_mat, 'cdat': c_mat}

    # initial parameter guesses
    pcateg = {c: np.zeros(s) for c, s in zip(c_names, Kl)}
    params = {'real': np.zeros(Kx), 'categ': pcateg, **extra}

    if hdfe is not None:
        params['hdfe'] = params['categ'].pop(hdfe)

    # estimate model
    beta, sigma = maxlike(model=model, params=params, data=data, stderr=stderr, disp=disp, **kwargs)

    if hdfe is not None:
        beta['categ'][hdfe] = beta.pop('hdfe')
        sigma['categ'][hdfe] = {'categ': {hdfe: sigma.pop('hdfe')}}

    return beta, sigma

# logit regression
logit_model = glm_model(link='logit', loss='binary')
def logit(y, x=[], fe=[], data=None, **kwargs):
    return glm(y, x=x, fe=fe, data=data, model=logit_model, **kwargs)

# poisson regression
poisson_model = glm_model(link='exp', loss='poisson')
def poisson(y, x=[], fe=[], data=None, **kwargs):
    return glm(y, x=x, fe=fe, data=data, model=poisson_model, **kwargs)

# zero inflated poisson regression
zinf_poisson_model = glm_model(link='exp', loss=zero_inflate(losses['poisson']))
def zinf_poisson(y, x=[], fe=[], data=None, **kwargs):
    extra = {'lpzero': 0.0}
    return glm(y, x=x, fe=fe, data=data, model=zinf_poisson_model, extra=extra, **kwargs)

# negative binomial regression
negbin_model = glm_model(link='exp', loss='negbin')
def negbin(y, x=[], fe=[], data=None, **kwargs):
    extra = {'lr': 0.0}
    return glm(y, x=x, fe=fe, data=data, model=negbin_model, extra=extra, **kwargs)

# zero inflated poisson regression
zinf_negbin_model = glm_model(link='exp', loss=zero_inflate(losses['negbin']))
def zinf_negbin(y, x=[], fe=[], data=None, **kwargs):
    extra = {'lpzero': 0.0, 'lr': 0.0}
    return glm(y, x=x, fe=fe, data=data, model=zinf_negbin_model, extra=extra, **kwargs)

# ordinary least squares (just for kicks)
def ols_loss(p, yh, y):
    lsigma = p['lsigma2']
    sigma2 = np.exp(lsigma2)
    like = -lsigma2 + lstsq_loss(yh, y)/sigma2
    return like

ols_model = glm_model(link='ident', loss=ols_loss)
def ols(y, x=[], fe=[], data=None, **kwargs):
    extra = {'lsigma': 0.0}
    return glm(y, x=x, fe=fe, data=data, model=ols_model, extra=extra, **kwargs)
