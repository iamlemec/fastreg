import pandas as pd
from operator import and_, add

import jax
import jax.numpy as np
import jax.numpy.linalg as la
from jax.scipy.special import gammaln
from jax.tree_util import tree_flatten, tree_leaves, tree_map, tree_reduce
from jax.lax import logistic
from jax.numpy import exp
import optax

from .tools import (
    block_inverse, chainer, maybe_diag, atleast_2d, hstack
)
from .formula import (
    parse_tuple, ensure_formula, categorize, is_categorical, Categ, Formula, O
)
from .trees import design_tree
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
        blike = np.clip(like0(p, d, yh, y), a_max=clip_like)
        zlike = log(pzero + (1-pzero)*exp(blike))
        plike = log(1-pzero) + blike
        return np.where(y == 0, zlike, plike)
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
    beta_reals = beta['reals']
    beta_categ = hstack(beta['categ'].values())

    sigma_reals = maybe_diag(sigma['reals']['reals']) if sigma['reals'] is not None else None
    sigma_categ = hstack([
        maybe_diag(sigma['categ'][c]['categ'][c]) for c in sigma['categ']
    ])

    beta_vec = hstack([beta_reals, beta_categ])
    sigma_vec = hstack([sigma_reals, sigma_categ])

    return beta_vec, sigma_vec

##
## optimizers
##

# figure out burn-in - cosine decay
def lr_schedule(eta, epochs, eta_boost=10.0, eta_burn=0.15):
    eta_burn = int(eta_burn*epochs) if type(eta_burn) is float else eta_burn
    def get_lr(ep):
        decay = np.clip(ep/eta_burn, 0, 1)
        coeff = 0.5*(1.0+np.cos(np.pi*decay))
        return eta*(1+coeff*(eta_boost-1))
    return get_lr

# adam optimizer with initial boost + cosine decay
def adam(
    vg_fun, loader, params0, epochs=10, eta=0.005, beta1=0.9, beta2=0.99, eps=1e-8,
    xtol=1e-4, ftol=1e-5, eta_boost=10.0, eta_burn=0.15, disp=None
):
    get_lr = lr_schedule(eta, epochs, eta_boost=eta_boost, eta_burn=eta_burn)

    # parameter info
    params = tree_map(np.array, params0)
    avg_loss = -np.inf

    # track rms gradient
    m = tree_map(np.zeros_like, params)
    v = tree_map(np.zeros_like, params)

    # do training
    for ep in range(epochs):
        # epoch stats
        agg_loss, agg_batch, tot_batch = 0.0, 0, 0
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
            m = tree_map(
                lambda m, g: beta1*m + (1-beta1)*g, m, grad
            )
            v = tree_map(
                lambda v, g: beta2*v + (1-beta2)*g**2, v, grad
            )

            # update with adjusted values
            lr = get_lr(ep)
            mhat = tree_map(lambda m: m/(1-beta1**(tot_batch+1)), m)
            vhat = tree_map(lambda v: v/(1-beta2**(tot_batch+1)), v)
            params = tree_map(
                lambda p, m, v: p + lr*m/(np.sqrt(v)+eps), params, mhat, vhat
            )

            # compute statistics
            agg_loss += loss
            agg_grad = tree_map(add, agg_grad, grad)
            agg_batch += 1
            tot_batch += 1

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

# adam optimizer with cosine burn in
def adam_cosine(learn=1e-2, boost=5.0, burn=0.2, epochs=None, **kwargs):
    burn = int(burn*epochs) if type(burn) is float else burn
    schedule = optax.cosine_decay_schedule(boost*learn, burn, alpha=1/boost)
    return optax.chain(
        optax.scale_by_adam(**kwargs),
        optax.scale_by_schedule(schedule)
    )

# adam optimizer with initial boost + cosine decay
def optax_wrap(
    vg_fun, loader, params0, optimizer=None, epochs=10, xtol=1e-4, ftol=1e-5,
    gtol=1e-4, disp=None, **kwargs
):
    # default optimizer
    if optimizer is None:
        optimizer = adam_cosine(epochs=epochs, **kwargs)

    # initialize optimizer
    params = tree_map(np.array, params0)
    state = optimizer.init(params)

    # start at bottom
    avg_loss = -np.inf

    # do training
    for ep in range(epochs):
        # epoch stats
        agg_loss, agg_batch, tot_batch = 0.0, 0, 0
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

            # update with adjusted values
            updates, state = optimizer.update(grad, state, params)
            params = optax.apply_updates(params, updates)

            # compute statistics
            agg_loss += loss
            agg_grad = tree_map(add, agg_grad, grad)
            agg_batch += 1
            tot_batch += 1

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
        if abs_grad < gtol and par_diff < xtol:
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
    model=None, params=None, data=None, stderr=False, optim=adam, batch_size=8192,
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
    model=None, params=None, data=None, vg_fun=None, stderr=True, optim=adam,
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
        ydat, rdat, cdat, odat = dat['ydat'], dat['rdat'], dat['cdat'], dat['odat']
        reals, categ = par['reals'], par['categ']
        if hdfe is not None:
            categ[hdfe] = par.pop('hdfe')

        # evaluate linear predictor
        pred = odat
        if rdat is not None:
            pred += rdat @ reals
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
    offset=None, model=None, loss=None, stderr=True, display=True, epochs=None,
    per=None, output='table', **kwargs
):
    # convert to formula system
    y, x = ensure_formula(x=x, y=y, formula=formula)

    # add in hdfe if needed
    if hdfe is not None:
        c_hdfe = parse_tuple(hdfe, convert=Categ)
        x += c_hdfe
        hdfe = c_hdfe.name()

    # add in raw data with offset special case
    if offset is None:
        offset = O

    # get all data in tree form
    formulify = lambda ts: Formula(*ts) if len(ts) > 0 else None
    c, r = map(formulify, categorize(is_categorical, x))
    tree = {'ydat': y, 'rdat': r, 'cdat': c, 'odat': offset, **raw}
    nam, dat = design_tree(tree, data=data)

    # handle no reals/categ case
    if tree['cdat'] is None:
        nam['cdat'] = {}
    if tree['rdat'] is None:
        nam['rdat'] = []

    # choose number of epochs
    N = len(dat['ydat'])
    epochs = max(1, 200_000_000 // N) if epochs is None else epochs
    per = max(1, epochs // 5) if per is None else per

    # create model if needed
    if model is None:
        model = glm_model(loss, hdfe=hdfe)

    # displayer
    def disp0(e, l, g, x, f, p, final=False):
        if e % per == 0 or final:
            reals, categ = p['reals'], p['categ']
            if hdfe is not None:
                categ = categ.copy()
                categ[hdfe] = p['hdfe']
            μr = np.mean(reals) if reals is not None else np.nan
            μc = np.mean(np.array([np.mean(c) for c in categ.values()]))
            print(f'[{e:3d}] ℓ={l:.5f}, g={g:.5f}, Δβ={x:.5f}, Δℓ={f:.5f}, μR={μr:.5f}, μC={μc:.5f}')
    disp = disp0 if display else None

    # organize data and initial params
    preals = np.zeros(len(nam['rdat'])) if len(nam['rdat']) > 0 else None
    pcateg = {c: np.zeros(len(ls)) for c, ls in nam['cdat'].items()}
    params = {'reals': preals, 'categ': pcateg, **extra}
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
        y_name = nam['ydat']
        x_names = nam['rdat'] + chainer(nam['cdat'].values())
        beta_vec, sigma_vec = flatten_output(beta, sigma)
        return param_table(beta_vec, y_name, x_names, sigma=sigma_vec)
    elif output == 'dict':
        names = {'reals': nam['rdat'], 'categ': nam['cdat']}
        return names, beta, sigma

# logit regression
def logit(y=None, x=None, data=None, **kwargs):
    return glm(y=y, x=x, data=data, loss='logit', **kwargs)

# poisson regression
def poisson(y=None, x=None, data=None, **kwargs):
    return glm(y=y, x=x, data=data, loss='poisson', **kwargs)

# zero inflated poisson regression
def poisson_zinf(y=None, x=None, data=None, clip_like=20.0, **kwargs):
    loss = zero_inflate(losses['poisson'], clip_like=clip_like)
    extra = {'lpzero': 0.0}
    return glm(y=y, x=x, data=data, loss=loss, extra=extra, **kwargs)

# negative binomial regression
def negbin(y=None, x=None, data=None, **kwargs):
    extra = {'lr': 0.0}
    return glm(y=y, x=x, data=data, loss='negbin', extra=extra, **kwargs)

# zero inflated poisson regression
def negbin_zinf(y=None, x=None, data=None, clip_like=20.0, **kwargs):
    loss = zero_inflate(losses['negbin'], clip_like=clip_like)
    extra = {'lpzero': 0.0, 'lr': 0.0}
    return glm(y=y, x=x, data=data, loss=loss, extra=extra, **kwargs)

# implement ols with full sigma
def gols(y=None, x=None, data=None, **kwargs):
    extra = {'lsigma2': 0.0}
    return glm(y=y, x=x, data=data, loss='normal', extra=extra, **kwargs)
