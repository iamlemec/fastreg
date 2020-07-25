import jax
import jax.lax as lax
import jax.scipy.special as spec
import jax.numpy as np
from jax.tree_util import tree_flatten, tree_leaves, tree_map, tree_multimap, tree_reduce
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

def todense(x):
    if sp.issparse(x):
        return x.todense()
    else:
        return x

class DataLoader:
    def __init__(self, data, batch_size):
        if type(data) is pd.DataFrame:
            data = data.to_dict('series')
        self.data = tree_map(np.array, data)
        self.batch_size = batch_size
        shapes = [d.shape[0] for d in tree_leaves(self.data)]
        self.data_size = shapes[0] # should all be the same size
        self.num_batches = max(1, self.data_size // batch_size)

    # if sparse, requires csc or csr format
    def __iter__(self):
        loc = 0
        for i in range(self.num_batches):
            yield tree_map(lambda d: todense(d[loc:loc+self.batch_size]), self.data)
            loc += self.batch_size

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

def get_sizes(dms, subset=None):
    if subset is None:
        subset = list(dms)
    return {k: v.size for k, v in dms.items() if k in subset}

def block_matrix(tree, dims, size):
    blocks = chunks(tree_leaves(tree), size)
    mat = np.block([[atleast_2d(x, axis=int(d>0)) for d, x in zip(dims, row)] for row in blocks])
    return mat

def block_unpack(mat, tree, sizes):
    part = np.cumsum(np.array(sizes))
    block = lambda x, axis: tree.unflatten(np.split(x, part, axis=axis)[:-1])
    tree = tree_map(lambda x: block(x, 1), block(mat, 0))
    return tree

##
## estimation
##

# maximum likelihood using jax - this expects a mean log likelihood
def maxlike(model, data, params, c={}, batch_size=8192, epochs=3, eta=0.01, gamma=0.9, disp=None, hessian=False, stderr=False):
    # compute derivatives if needed
    vg_fun = jax.jit(jax.value_and_grad(model))

    # construct dataset
    loader = DataLoader(data, batch_size)
    N, B = loader.data_size, loader.num_batches

    # parameter info
    params = tree_map(np.array, params)
    par_flat, par_tree = tree_flatten(params)
    par_sizes = [len(p) if p.ndim > 0 else 1 for p in par_flat]
    par_dims = [np.ndim(p) for p in par_flat]
    K = len(par_flat)

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

    if not hessian:
        return params

    # get hessian matrix
    if hessian == 'deriv':
        h_fun = jax.jit(jax.hessian(model))
        hess = None
        for b, batch in enumerate(loader):
            h_batch = h_fun(params, batch)
            if hess is None:
                hess = h_batch
            else:
                hess = tree_multimap(np.add, hess, h_batch)
        hess = tree_map(lambda x: x/B, hess)
    elif hessian == 'outer':
        tree_axis = tree_map(lambda _: 0, data)
        vg_fun = jax.jit(jax.vmap(jax.grad(model), (None, tree_axis), 0))
        hess = None
        for b, batch in enumerate(loader):
            vg_batch = vg_fun(params, batch)
            gg_batch = tree_map(lambda x: tree_map(lambda y: x.T @ y, vg_batch), vg_batch)
            if hess is None:
                hess = gg_batch
            else:
                hess = tree_multimap(np.add, hess, gg_batch)
        hess = tree_map(lambda x: -x/N, hess)

    if not stderr:
        return params, hess

    # invert hessian for stderrs
    hmat = block_matrix(hess, par_dims, K)
    ifish = -inv_fun(hmat)/N
    sigma = block_unpack(ifish, par_tree, par_sizes)

    # full standard errors
    return params, sigma

# make a glm model and compile
def glm_model(link, loss, full=True):
    if type(link) is str:
        link = links[link]
    if type(loss) is str:
        loss = losses[loss]

    # evaluator
    def model(par, dat):
        ydat, xdat, cdat = dat['ydat'], dat['xdat'], dat['cdat']
        real, categ = par['real'], par['categ']
        linear = xdat @ real
        for i, c in enumerate(categ):
            cidx = cdat.T[i] # needed for vmap to work
            linear += categ[c][cidx]
        pred = link(linear)
        like = loss(par, pred, ydat)
        return np.mean(like)

    return model

# default glm specification
def glm(y, x=[], fe=[], data=None, extra={}, model=None, link=None, loss=None, intercept=True, stderr=True, **kwargs):
    # construct design matrices
    y_vec, x_mat, x_names, c_mat, c_names = design_matrices(y, x=x, fe=fe, data=data, intercept=intercept, method='ordinal')

    # get data shape
    N = len(y_vec)
    Kx = len(x_names)
    Kc = len(c_names)
    Kl = [len(c) for c in c_names.values()]

    # compile model if needed
    if model is None:
        model = glm_model(link, loss)

    # displayer
    def disp(e, l, p):
        real, categ = p['real'], p['categ']
        mcats = np.array([np.mean(c) for c in categ.values()])
        print(f'[{e:3d}] {l:.4f}: {np.mean(real):.4f} {np.mean(mcats):.4f}')

    # estimate model
    data = {'ydat': y_vec, 'xdat': x_mat, 'cdat': c_mat}
    pcateg0 = {c: np.zeros(s) for c, s in zip(c_names, Kl)}
    params0 = {'real': np.zeros(Kx), 'categ': pcateg0, **extra}
    beta, sigma = maxlike(model, data, params0, data, stderr=stderr, disp=disp, **kwargs)

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
