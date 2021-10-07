import jax
import jax.lax as lax
from jax.scipy.special import gammaln
import jax.numpy as np
import jax.numpy.linalg as la
from jax.tree_util import (
    tree_flatten, tree_leaves, tree_map, tree_reduce, tree_structure
)
from jax.interpreters.xla import DeviceArray
import numpy as np0
import scipy.sparse as sp
import pandas as pd
from operator import and_, add

from .formula import (
    design_matrices, parse_item, parse_tuple, parse_list, ensure_formula, Categ
)
from .tools import block_inverse, chainer, maybe_diag
from .summary import param_table

##
## constants
##

# numbers
eps = 1e-7

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
def zero_inflate(like0, clip_like=20.0, key='lpzero'):
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
        self.data = tree_map(
            lambda x: np.array(x) if type(x) is not np.ndarray else x, data
        )
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

class OneLoader:
    def __init__(self, data, batch_size=None):
        self.data = data

    def __iter__(self):
        yield from self.iterate()

    def iterate(self):
        yield self.data

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
    mat = np.block([
        [atleast_2d(x, axis=int(d>0)) for d, x in zip(dims, row)]
        for row in blocks
    ])
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
            total = tree_map(agg_fun, total, f_batch)
    return total

def tree_outer(tree):
    return tree_map(lambda x: tree_map(lambda y: x.T @ y, tree), tree)

def tree_outer_flat(tree):
    tree1, vec = dict_popoff(tree, 'hdfe')
    leaves = [np.atleast_2d(l.T).T for l in tree_leaves(tree1)]
    mat = np.hstack(leaves)
    A = mat.T @ mat
    B = mat.T @ vec
    C = vec.T @ mat
    d = np.sum(vec*vec, axis=0)
    return A, B, C, d

def dict_popoff(d, s):
    if s in d:
        return {k: v for k, v in d.items() if k != s}, d[s]
    else:
        return d, None

def tree_fisher(vg_fun, params, loader):
    # accumulate outer product
    fish = tree_batch_reduce(
        lambda b: tree_outer(vg_fun(params, b)), loader
    )

    # invert fisher matrix
    sigma = tree_matfun(la.inv, fish, params)

    return sigma

def diag_fisher(vg_fun, params, loader):
    # compute hessian inverse by block
    A, B, C, d = tree_batch_reduce(
        lambda b: tree_outer_flat(vg_fun(params, b)), loader
    )
    psig, hsig = block_inverse(A, B, C, d, inv=la.inv)

    # unpack into tree
    par0, _ = dict_popoff(params, 'hdfe')
    par0_flat, par0_tree = tree_flatten(par0)
    par0_sizs = [np.size(p) for p in par0_flat]
    sigma = block_unpack(psig, par0_tree, par0_sizs)
    sigma['hdfe'] = hsig

    return sigma

# just get mean and var vectors
def flatten_output(beta, sigma):
    beta_real = beta['real']
    beta_categ = np.hstack([
        beta['categ'][c] for c in beta['categ']
    ])

    sigma_real = maybe_diag(sigma['real']['real'])
    sigma_categ = np.hstack([
        maybe_diag(sigma['categ'][c]['categ'][c]) for c in sigma['categ']
    ])

    beta_vec = np.hstack([beta_real, beta_categ])
    sigma_vec = np.hstack([sigma_real, sigma_categ])

    return beta_vec, sigma_vec

##
## optimizers
##

def adam(vg_fun, loader, params0, epochs=10, eta=0.01, gamma=0.9, disp=None):
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
            gnan = tree_reduce(
                and_, tree_map(lambda g: np.isnan(g).any(), grad)
            )

            if lnan or gnan:
                print('Encountered nans!')
                return params, None

            grms = tree_map(
                lambda r, g: gamma*r + (1-gamma)*g**2, grms, grad
            )
            params = tree_map(
                lambda p, g, r: p + eta*g/np.sqrt(r+eps), params, grad, grms
            )

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

# maximum likelihood using jax - this expects a mean log likelihood
def maxlike(
    model=None, params=None, data=None, stderr=False, optim=adam, backend='gpu',
    **kwargs
):
    # get model gradients
    vg_fun = jax.jit(jax.value_and_grad(model), backend=backend)

    # simple non-batched loader
    loader = OneLoader(data)

    # maximize likelihood
    params1 = optim(vg_fun, loader, params, **kwargs)

    if not stderr:
        return params1, None

    # get model hessian
    h_fun = jax.jit(jax.hessian(model), backend=backend)

    # compute standard errors
    hess = h_fun(params, data)
    fish = tree_matfun(inv_fun, hess, params)
    omega = tree_map(lambda x: -x, fish)

    return params1, omega

# maximum likelihood using jax - this expects a mean log likelihood
# the assumes the data is batchable, which usually means panel-like
# a toplevel hdfe variable is treated special-like
def maxlike_panel(
    model=None, params=None, data=None, vg_fun=None, stderr=True, optim=adam,
    batch_size=8192, batch_stderr=8192, backend='gpu', **kwargs
):
    # compute gradient for optim
    vg_fun = jax.jit(jax.value_and_grad(model), backend=backend)

    # set up batching
    loader = DataLoader(data, batch_size)

    # maximize likelihood
    params1 = optim(vg_fun, loader, params, **kwargs)

    # just point estimates
    if not stderr:
        return params1, None

    # get vectorized gradient
    gv_fun = jax.jit(jax.vmap(jax.grad(model), (None, 0), 0), backend=backend)

    # batching for stderr
    hload = loader(batch_stderr)

    # compute standard errors
    if 'hdfe' in params:
        sigma = diag_fisher(gv_fun, params1, loader)
    else:
        sigma = tree_fisher(gv_fun, params1, loader)

    return params1, sigma

# make a glm model and compile
def glm_model(link, loss, hdfe=None, drop='first'):
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
            if drop == 'first':
                linear += np.where(cidx > 0, categ[c][cidx-1], 0.0)
            else:
                linear += categ[c][cidx]
        pred = link(linear)
        like = loss(par, pred, ydat)
        return np.mean(like)

    return model

# default glm specification
def glm(
    y=None, x=None, formula=None, hdfe=None, data=None, extra={}, model=None,
    link=None, loss=None, extern=None, stderr=True, drop='first', display=True,
    epochs=None, per=None, output='table', **kwargs
):
    # convert to formula system
    y, x = ensure_formula(x=x, y=y, formula=formula)

    # add in hdfe if needed
    if hdfe is not None:
        c_hdfe = parse_tuple(hdfe, convert=Categ)
        x += c_hdfe
        hdfe = c_hdfe.name()

    # construct design matrices
    y_vec, y_name, x_mat, x_names, c_mat, c_names0 = design_matrices(
        y=y, x=x, data=data, method='ordinal', extern=extern
    )

    # accumulate all names
    if drop == 'first':
        c_names0 = {k: v[1:] for k, v in c_names0.items()}
    c_names = chainer(c_names0.values())
    names = x_names + c_names

    # get data shape
    N = len(y_vec)
    Kx = len(x_names)
    Kc = len(c_names0)
    Kl = [len(c) for c in c_names0.values()]

    # choose number of epochs
    epochs = max(1, 2_000_000 // N) if epochs is None else epochs
    per = max(1, epochs // 5) if per is None else per

    # compile model if needed
    if model is None:
        model = glm_model(link, loss, hdfe=hdfe, drop=drop)

    # displayer
    def disp(e, l, p):
        real, categ = p['real'], p['categ']
        if hdfe is not None:
            categ = categ.copy()
            categ[hdfe] = p['hdfe']
        mcats = np.array([np.mean(c) for c in categ.values()])
        if e % per == 0:
            print(f'[{e:3d}] {l:.5f}: {np.mean(real):.5f} {np.mean(mcats):.5f}')
    disp1 = disp if display else None

    # organize data and initial params
    data = {'ydat': y_vec, 'xdat': x_mat, 'cdat': c_mat}
    pcateg = {c.name(): np.zeros(s) for c, s in zip(c_names0, Kl)}
    params = {'real': np.zeros(Kx), 'categ': pcateg, **extra}
    if hdfe is not None:
        params['hdfe'] = params['categ'].pop(hdfe)

    # estimate model
    beta, sigma = maxlike_panel(
        model=model, params=params, data=data, stderr=stderr, disp=disp1,
        epochs=epochs, **kwargs
    )

    if hdfe is not None:
        beta['categ'][hdfe] = beta.pop('hdfe')
        if stderr:
            sigma['categ'][hdfe] = {'categ': {hdfe: sigma.pop('hdfe')}}

    if output == 'table':
        beta_vec, sigma_vec = flatten_output(beta, sigma)
        return param_table(beta_vec, sigma_vec, y_name, names)
    elif output == 'dict':
        return {
            'beta': beta,
            'sigma': sigma,
        }

# logit regression
def logit(y=None, x=None, data=None, **kwargs):
    return glm(y=y, x=x, data=data, link='logit', loss='binary', **kwargs)

# poisson regression
def poisson(y=None, x=None, data=None, **kwargs):
    return glm(y=y, x=x, data=data, link='exp', loss='poisson', **kwargs)

# zero inflated poisson regression
def zinf_poisson(y=None, x=None, data=None, clip_like=20.0, **kwargs):
    return glm(
        y=y, x=x, data=data, link='exp',
        loss=zero_inflate(losses['poisson'], clip_like=clip_like),
        extra={'lpzero': 0.0}, **kwargs
    )

# negative binomial regression
def negbin(y=None, x=None, data=None, **kwargs):
    return glm(
        y=y, x=x, data=data, link='exp', loss='negbin', extra={'lr': 0.0},
        **kwargs
    )

# zero inflated poisson regression
def zinf_negbin(y=None, x=None, data=None, clip_like=20.0, **kwargs):
    return glm(
        y=y, x=x, data=data, link='exp',
        loss=zero_inflate(losses['negbin'], clip_like=clip_like),
        extra={'lpzero': 0.0, 'lr': 0.0}, **kwargs
    )

# ordinary least squares (just for kicks)
def ols_loss(p, yh, y):
    lsigma2 = p['lsigma2']
    sigma2 = np.exp(lsigma2)
    like = -lsigma2 + lstsq_loss(yh, y)/sigma2
    return like

def gols(y=None, x=None, data=None, **kwargs):
    return glm(
        y=y, x=x, data=data, link='ident', loss=ols_loss,
        extra={'lsigma2': 0.0}, **kwargs
    )
