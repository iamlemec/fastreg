import pandas as pd
from operator import and_, add

import jax
import jax.numpy as np
import jax.numpy.linalg as la
from jax.scipy.special import gammaln
from jax.tree_util import tree_flatten, tree_leaves, tree_map, tree_reduce
from jax.lax import logistic
from jax.numpy import exp

from .formula import (
    design_matrices, parse_tuple, ensure_formula, parse_item, Categ
)
from .tools import (
    block_inverse, chainer, maybe_diag, atleast_2d, hstack, valid_rows, all_valid
)
from .summary import param_table

##
## basic functions
##

# clamped log
def log(x, ε=1e-7):
    return np.log(np.maximum(ε, x))

##
## canned models
##

# loss functions (only parameter relevant terms)
def binary_loss(yh, y):
    return y*log(yh) + (1-y)*log(1-yh)

def poisson_loss(yh, y):
    return y*log(yh) - yh

def negbin_loss(r, yh, y):
    return gammaln(r+y) - gammaln(r) + r*log(r) + y*log(yh) - (r+y)*log(r+yh)

def lstsq_loss(yh, y):
    return -(y-yh)**2

def normal_loss(p, yh, y):
    lsigma2 = p['lsigma2']
    sigma2 = np.exp(lsigma2)
    like = -0.5*lsigma2 + 0.5*lstsq_loss(yh, y)/sigma2
    return like

losses = {
    'logit': lambda p, d, yh, y: binary_loss(logistic(yh), y),
    'poisson': lambda p, d, yh, y: poisson_loss(exp(yh), y),
    'negbin': lambda p, d, yh, y: negbin_loss(exp(p['lr']), exp(yh), y),
    'normal': lambda p, d, yh, y: normal_loss(p, yh, y),
    'lognorm': lambda p, d, yh, y: normal_loss(p, yh, log(y)),
    'lstsq': lambda p, d, yh, y: lstsq_loss(yh, y),
}

def ensure_loss(s):
    if type(s) is str:
        return losses[s]
    else:
        return s

# loss function modifiers
def zero_inflate(like0, clip_like=20.0, key='lpzero'):
    like0 = ensure_loss(like0)
    def like(p, d, yh, y):
        pzero = logistic(p[key])
        plike = np.clip(like0(p, d, yh, y), a_max=clip_like)
        llike = np.where(y == 0, log(pzero), log(1-pzero) + plike)
        return llike
    return like

def add_offset(like0, key='offset'):
    like0 = ensure_loss(like0)
    def like(p, d, yh, y):
        yh1 = d[key] + yh
        return like0(p, d, yh1, y)
    return like

##
## batching it, pytree style
##

class DataLoader:
    def __init__(self, data, batch_size=None):
        # robust input handling
        if type(data) is pd.DataFrame:
            data = data.to_dict('series')

        # note that tree_map seems to drop None valued leaves
        self.data = tree_map(
            lambda x: np.array(x) if type(x) is not np.ndarray else x, data
        )

        # validate shapes
        shapes = set([d.shape[0] for d in tree_leaves(self.data)])
        if len(shapes) > 1:
            raise Exception('All data series must have first dimension size')

        # store for iteration
        self.data_size, = shapes
        self.batch_size = batch_size

    def __call__(self, batch_size=None):
        yield from self.iterate(batch_size)

    def __iter__(self):
        yield from self.iterate()

    def iterate(self, batch_size=None):
        # round off data size to batch_size multiple
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_batches = max(1, self.data_size // batch_size)
        round_size = batch_size*num_batches

        # yield successive tree batches
        for i in range(0, round_size, batch_size):
            yield tree_map(lambda d: d[i:i+batch_size], self.data)

# ignore batch_size and use entire dataset
class OneLoader:
    def __init__(self, data, batch_size=None):
        self.data = data

    def __call__(self, batch_size=None):
        yield from self.iterate()

    def __iter__(self):
        yield from self.iterate()

    def iterate(self, batch_size=None):
        yield self.data

##
## block matrices
##

def chunks(v, n):
    return [v[i:i+n] for i in range(0, len(v), n)]

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
    leaves = [atleast_2d(l) for l in tree_leaves(tree1)]
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

def tree_fisher(gv_fun, params, loader):
    # accumulate outer product
    fish = tree_batch_reduce(
        lambda b: tree_outer(gv_fun(params, b)), loader
    )

    # invert fisher matrix
    sigma = tree_matfun(la.inv, fish, params)

    return sigma

def diag_fisher(gv_fun, params, loader):
    # compute hessian inverse by block
    A, B, C, d = tree_batch_reduce(
        lambda b: tree_outer_flat(gv_fun(params, b)), loader
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
    beta_categ = hstack([
        beta['categ'][c] for c in beta['categ']
    ])

    sigma_real = maybe_diag(sigma['real']['real'])
    sigma_categ = hstack([
        maybe_diag(sigma['categ'][c]['categ'][c]) for c in sigma['categ']
    ])

    beta_vec = hstack([beta_real, beta_categ])
    sigma_vec = hstack([sigma_real, sigma_categ])

    return beta_vec, sigma_vec

##
## optimizers
##

def rmsprop(
    vg_fun, loader, params0, epochs=10, eta=0.001, gamma=0.9, eps=1e-7,
    xtol=1e-3, ftol=1e-5, disp=None
):
    # parameter info
    params = tree_map(np.array, params0)
    avg_loss = -np.inf

    # track rms gradient
    grms = tree_map(np.zeros_like, params)

    # do training
    for ep in range(epochs):
        # epoch stats
        agg_loss, agg_batch = 0.0, 0
        agg_grad = tree_map(np.zeros_like, params0)
        last_par, last_loss = params, avg_loss

        # iterate over batches
        for batch in loader:
            # compute gradients
            loss, grad = vg_fun(params, batch)

            # check for any nans
            lnan = np.isnan(loss)
            gnan = tree_reduce(
                and_, tree_map(lambda g: np.isnan(g).any(), grad)
            )
            if lnan or gnan:
                print('Encountered nans!')
                return params, None

            # implement next step
            grms = tree_map(
                lambda r, g: gamma*r + (1-gamma)*g**2, grms, grad
            )
            params = tree_map(
                lambda p, g, r: p + eta*g/np.sqrt(r+eps), params, grad, grms
            )

            # compute statistics
            agg_loss += loss
            agg_grad = tree_map(add, agg_grad, grad)
            agg_batch += 1

        # compute stats
        avg_loss = agg_loss/agg_batch
        avg_grad = tree_map(lambda x: x/agg_batch, agg_grad)
        abs_grad = tree_reduce(np.maximum, tree_map(lambda x: np.max(np.abs(x)), avg_grad))
        par_diff = tree_reduce(
            np.maximum, tree_map(lambda p1, p2: np.max(np.abs(p1-p2)), params, last_par)
        )
        loss_diff = np.abs(avg_loss-last_loss)

        # display output
        if disp is not None:
            disp(ep, avg_loss, abs_grad, par_diff, loss_diff, params)

        # check converge
        if par_diff < xtol and loss_diff < ftol:
            break

    # show final result
    if disp is not None:
        disp(ep, avg_loss, abs_grad, par_diff, loss_diff, params, final=True)

    return params

##
## estimation
##

# maximum likelihood using jax - this expects a mean log likelihood
def maxlike(
    model=None, params=None, data=None, stderr=False, optim=rmsprop, batch_size=8192,
    backend='cpu', **kwargs
):
    # get model gradients
    vg_fun = jax.jit(jax.value_and_grad(model), backend=backend)

    # simple non-batched loader
    BatchLoader = OneLoader if batch_size is None else DataLoader
    loader = BatchLoader(data)

    # maximize likelihood
    params1 = optim(vg_fun, loader, params, **kwargs)

    if not stderr:
        return params1, None

    # get model hessian
    h_fun = jax.jit(jax.hessian(model), backend=backend)

    # compute standard errors
    hess = tree_batch_reduce(lambda b: h_fun(params, b), loader)
    fish = tree_matfun(np.linalg.inv, hess, params)
    omega = tree_map(lambda x: -x, fish)

    return params1, omega

# maximum likelihood using jax - this expects a mean log likelihood
# the assumes the data is batchable, which usually means panel-like
# a toplevel hdfe variable is treated special-like in diag_fisher
def maxlike_panel(
    model=None, params=None, data=None, vg_fun=None, stderr=True, optim=rmsprop,
    batch_size=8192, backend='cpu', **kwargs
):
    # compute gradient for optim
    vg_fun = jax.jit(jax.value_and_grad(model), backend=backend)

    # set up batching
    BatchLoader = OneLoader if batch_size is None else DataLoader
    loader = BatchLoader(data, batch_size)

    # maximize likelihood
    params1 = optim(vg_fun, loader, params, **kwargs)

    # just point estimates
    if not stderr:
        return params1, None

    # get vectorized gradient
    gv_fun = jax.jit(jax.vmap(jax.grad(model), (None, 0), 0), backend=backend)

    # compute standard errors
    if 'hdfe' in params:
        sigma = diag_fisher(gv_fun, params1, loader)
    else:
        sigma = tree_fisher(gv_fun, params1, loader)

    return params1, sigma

# make a glm model with a particular loss
def glm_model(loss, hdfe=None):
    if type(loss) is str:
        loss = losses[loss]

    # evaluator function
    def model(par, dat):
        # load in data and params
        ydat, xdat, cdat = dat['ydat'], dat['xdat'], dat['cdat']
        real, categ = par['real'], par['categ']
        if hdfe is not None:
            categ[hdfe] = par.pop('hdfe')

        # evaluate linear predictor
        pred = xdat @ real
        for i, c in enumerate(categ):
            cidx = cdat.T[i] # needed for vmap to work
            pred += np.where(cidx >= 0, categ[c][cidx], 0.0) # -1 means drop

        # compute average likelihood
        like = loss(par, dat, pred, ydat)
        return np.mean(like)

    return model

# default glm specification
def glm(
    y=None, x=None, formula=None, hdfe=None, data=None, extra={}, raw={},
    offset=None, model=None, loss=None, extern=None, stderr=True, display=True,
    epochs=None, per=None, output='table', **kwargs
):
    # convert to formula system
    y, x = ensure_formula(x=x, y=y, formula=formula)

    # add in hdfe if needed
    if hdfe is not None:
        c_hdfe = parse_tuple(hdfe, convert=Categ)
        x += c_hdfe
        hdfe = c_hdfe.name()

    # add in raw data with offset special case
    if offset is not None:
        raw = {**raw, 'offset': offset}
    r_vec = {k: parse_item(v).raw(data, extern=extern) for k, v in raw.items()}
    r_val = all_valid(*[valid_rows(v) for v in r_vec.values()])

    # construct design matrices
    y_vec, y_name, x_mat, x_names, c_mat, c_names, valid = design_matrices(
        y=y, x=x, data=data, method='ordinal', extern=extern, flatten=False,
        validate=True, valid0=r_val
    )

    # drop invalid raw rows
    r_vec = {k: v[valid] for k, v in r_vec.items()}

    # accumulate all names
    c_names = {c.name(): ls for c, ls in c_names.items()}

    # get data shape
    N = len(y_vec)
    Kx = len(x_names)

    # choose number of epochs
    epochs = max(1, 50_000_000 // N) if epochs is None else epochs
    per = max(1, epochs // 5) if per is None else per

    # create model if needed
    if model is None:
        if offset is not None:
            loss = add_offset(loss, key='offset')
        model = glm_model(loss, hdfe=hdfe)

    # displayer
    def disp0(e, l, g, x, f, p, final=False):
        real, categ = p['real'], p['categ']
        if hdfe is not None:
            categ = categ.copy()
            categ[hdfe] = p['hdfe']
        mcats = np.array([np.mean(c) for c in categ.values()])
        if e % per == 0 or final:
            μR, μC = np.mean(real), np.mean(mcats)
            print(f'[{e:3d}] ℓ={l:.5f}, g={g:.5f}, Δβ={x:.5f}, Δℓ={f:.5f}, μR={μR:.5f}, μC={μC:.5f}')
    disp = disp0 if display else None

    # organize data and initial params
    dat = {'ydat': y_vec, 'xdat': x_mat, 'cdat': c_mat, **r_vec}
    pcateg = {c: np.zeros(len(ls)) for c, ls in c_names.items()}
    params = {'real': np.zeros(Kx), 'categ': pcateg, **extra}
    if hdfe is not None:
        params['hdfe'] = params['categ'].pop(hdfe)

    # estimate model
    beta, sigma = maxlike_panel(
        model=model, params=params, data=dat, stderr=stderr, disp=disp,
        epochs=epochs, **kwargs
    )

    # splice in hdfe results
    if hdfe is not None:
        beta['categ'][hdfe] = beta.pop('hdfe')
        if stderr:
            sigma['categ'][hdfe] = {'categ': {hdfe: sigma.pop('hdfe')}}

    # return requested info
    if output == 'table':
        names = x_names + chainer(c_names.values())
        beta_vec, sigma_vec = flatten_output(beta, sigma)
        return param_table(beta_vec, y_name, names, sigma=sigma_vec)
    elif output == 'dict':
        names = {'real': x_names, 'categ': c_names}
        return names, beta, sigma

# logit regression
def logit(y=None, x=None, data=None, **kwargs):
    return glm(y=y, x=x, data=data, loss='logit', **kwargs)

# poisson regression
def poisson(y=None, x=None, data=None, **kwargs):
    return glm(y=y, x=x, data=data, loss='poisson', **kwargs)

# zero inflated poisson regression
def zinf_poisson(y=None, x=None, data=None, clip_like=20.0, **kwargs):
    loss = zero_inflate(losses['poisson'], clip_like=clip_like)
    extra = {'lpzero': 0.0}
    return glm(y=y, x=x, data=data, loss=loss, extra=extra, **kwargs)

# negative binomial regression
def negbin(y=None, x=None, data=None, **kwargs):
    extra = {'lr': 0.0}
    return glm(y=y, x=x, data=data, loss='negbin', extra=extra, **kwargs)

# zero inflated poisson regression
def zinf_negbin(y=None, x=None, data=None, clip_like=20.0, **kwargs):
    loss = zero_inflate(losses['negbin'], clip_like=clip_like)
    extra = {'lpzero': 0.0, 'lr': 0.0}
    return glm(y=y, x=x, data=data, loss=loss, extra=extra, **kwargs)

# implement ols with full sigma
def gols(y=None, x=None, data=None, **kwargs):
    extra = {'lsigma2': 0.0}
    return glm(y=y, x=x, data=data, loss='normal', extra=extra, **kwargs)
