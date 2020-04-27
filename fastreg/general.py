import jax
import jax.lax as lax
import jax.scipy.special as spec
import jax.numpy as np
from jax.tree_util import tree_leaves, tree_map, tree_multimap, tree_reduce
from jax.interpreters.xla import DeviceArray
import numpy as np0
import scipy.sparse as sp
import pandas as pd

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

def sigmoid(x):
    return 1/(1+np.exp(-x))

def log(x):
    return np.log(np.maximum(eps, x))

# link functions
links = {
    'identity': lambda x: x,
    'exponential': lambda x: np.exp(x),
    'logit': lambda x: 1/(1+np.exp(-x))
}

# loss functions
losses = {
    'binary': lambda yh, y: y*log(yh) + (1-y)*log(1-yh),
    'poisson': lambda yh, y: y*log(yh) - yh,
    'negative_binomial': lambda r, yh, y: gammaln(r+y) - gammaln(r) + r*log(r) + y*log(yh) - (r+y)*log(r+yh),
    'least_squares': lambda yh, y: -(y-yh)**2
}

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

def atleast_2d(x, axis=0):
    x2 = np.atleast_2d(x)
    if x2.shape[axis] == 1:
        x2 = x2.T
    return x2

def get_sizes(dms, subset=None):
    if subset is None:
        subset = list(dms)
    return {k: v.size for k, v in dms.items() if k in subset}

def block_select(dms, sel1=None, sel2=None):
    if sel1 is None:
        sel1 = list(dms)
    if sel2 is None:
        sel2 = list(dms)
    return np.block([[atleast_2d(dms[k1][k2], axis=int(i2 < i1)) for i2, k2 in enumerate(dms) if k2 in sel1] for i1, k1 in enumerate(dms) if k1 in sel1])

def block_unpack(mat, sizes):
    out = {k: {} for k in sizes}
    pos = [0] + np0.cumsum(list(sizes.values())).tolist()
    for i1, k1 in enumerate(sizes):
        for i2, k2 in enumerate(sizes):
            out[k1][k2] = np.squeeze(mat[pos[i1]:pos[i1+1],pos[i2]:pos[i2+1]])
    return out

##
## estimation
##

# maximum likelihood using jax - this expects a mean log likelihood
def maxlike(model, data, params, c={}, batch_size=4092, epochs=3, learning_rate=0.5, eta=0.001, gamma=0.9, per=100, disp=None, stderr=True):
    # compute derivatives
    vg_fun = jax.jit(jax.value_and_grad(model))
    g_fun = jax.jit(jax.grad(model))
    h_fun = jax.jit(jax.hessian(model))

    # construct dataset
    data = DataLoader(data, batch_size)
    N = data.data_size

    # track rms gradient
    grms = tree_map(np.zeros_like, params)

    # do training
    for ep in range(epochs):
        # epoch stats
        agg_loss, agg_batch = 0.0, 0

        # iterate over batches
        for b, batch in enumerate(data):
            # compute gradients
            loss, grad = vg_fun(params, batch)

            lnan = np.isnan(loss)
            pnan = tree_map(lambda g: np.isnan(g).any(), grad)
            gnan = tree_reduce(lambda a, b: a & b, pnan)

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

    return params, None

    # get hessian matrix (should use pytree)
    shapes = {k: v.shape for k, v in params.items()}
    hess = {k1: {k2: np.zeros(v1+v2) for k2, v2 in shapes.items()} for k1, v1 in shapes.items()}
    for y_bat, x_bat in data:
        h = h_fun(params, y_bat, x_bat)
        for k1 in h:
            for k2 in h:
                hess[k1][k2] += h[k1][k2]*(batch_size/N)

    # get cov matrix
    if stderr is True:
        fish = block_select(hess)
        ifish = -np.linalg.inv(fish)/N
        sizes = get_sizes(params)
        sigma = block_unpack(ifish, sizes)
    elif stderr is None:
        sigma = N*hess
    else:
        # use block diagonal inverse formula
        # https://en.wikipedia.org/wiki/Invertible_matrix#Blockwise_inversion
        nocomp = [k for k in params if k not in stderr]

        A = block_select(hess, stderr, stderr)
        B = block_select(hess, stderr, nocomp)
        C = block_select(hess, nocomp, stderr)
        D = block_select(hess, nocomp, nocomp)

        A1 = np.linalg.inv(A)
        Z1 = np.linalg.inv(D - C @ A1 @ B)
        I1 = A1 + A1 @ B @ Z1 @ C @ A1
        ifish = -I1/N

        sizes = get_sizes(params, subset=stderr)
        sigma = block_unpack(ifish, sizes)

    # return to device
    return params, sigma

# default glm specification
def glm(y, x=[], fe=[], data=None, extra={}, link=None, loss=None, intercept=True, stderr=True, **kwargs):
    # construct design matrices
    y_vec, x_mat, x_names, c_mat, c_names = design_matrices(y, x=x, fe=fe, data=data, intercept=intercept, method='ordinal')

    # get data shape
    N = len(y_vec)
    Kx = len(x_names)
    Kc = len(c_names)
    Kl = [len(c) for c in c_names.values()]

    # evaluator
    def model(par, dat):
        ydat, xdat, cdat = dat['ydat'], dat['xdat'], dat['cdat']
        beta_x, beta_c = par['beta_x'], par['beta_c']
        linear = xdat @ beta_x
        for i, c in enumerate(beta_c):
            linear += beta_c[c][cdat[:, i]]
        pred = link(linear)
        like = loss(par, pred, ydat)
        return np.mean(like)

    # displayer
    def disp(e, l, p):
        beta_x, beta_c = p['beta_x'], p['beta_c']
        c_means = np.array([np.mean(c) for c in beta_c.values()])
        print(f'[{e:3d}] {l:.4f}: {np.mean(beta_x):.4f} {np.mean(c_means):.4f}')

    # estimate model
    data = {'ydat': y_vec, 'xdat': x_mat, 'cdat': c_mat}
    pcateg0 = {c: np.zeros(s) for c, s in zip(c_names, Kl)}
    params0 = {'beta_x': np.zeros(Kx), 'beta_c': pcateg0, **extra}
    params, sigma = maxlike(model, data, params0, data, stderr=stderr, disp=disp, **kwargs)

    # return relevant
    return params, sigma

# logit regression
def logit(y, x=[], fe=[], data=None, **kwargs):
    link = links['logit']
    like0 = losses['binary']
    like = lambda p, yh, y: like0(yh, y)
    return glm(y, x=x, fe=fe, data=data, link=link, loss=like, **kwargs)

# poisson regression
def poisson(y, x=[], fe=[], data=None, **kwargs):
    link = links['exponential']
    like0 = losses['poisson']
    like = lambda p, yh, y: like0(yh, y)
    return glm(y, x=x, fe=fe, data=data, link=link, loss=like, **kwargs)

# zero inflated poisson regression
def zero_inflated_poisson(y, x=[], fe=[], data=None, **kwargs):
    # base poisson distribution
    link = links['exponential']
    like0 = losses['poisson']
    extra = {'lpzero': 0.0}

    # zero inflation
    def loss(par, yh, y):
        pzero = sigmoid(par['lpzero'])
        clike = np.clip(like0(yh, y), a_max=clip_like)
        like = pzero*(y==0) + (1-pzero)*np.exp(clike)
        return log(like)

    # pass to glm
    par, sig = glm(y, x=x, fe=fe, data=data, extra=extra, link=link, loss=loss, **kwargs)

    # transform params
    par['pzero'] = sigmoid(par.pop('lpzero'))

    return par, sig

# negative binomial regression
def negative_binomial(y, x=[], fe=[], data=None, **kwargs):
    link = links['exponential']
    like = losses['negative_binomial']
    extra = {'lalpha': 0.0}

    def loss(par, yh, y):
        r = np.exp(-par['lalpha'])
        return like(r, yh, y)

    # pass to glm
    par = glm(y, x=x, fe=fe, data=data, extra=extra, link=link, loss=loss, **kwargs)

    # transform params
    par['alpha'] = np.exp(-par.pop('lalpha'))

    return par

# zero inflated poisson regression
def zero_inflated_negative_binomial(y, x=[], fe=[], data=None, **kwargs):
    # base poisson distribution
    link = links['exponential']
    like0 = losses['negative_binomial']
    extra = {
        'lpzero': 0.0,
        'lalpha': 0.0
    }

    # zero inflation
    def loss(par, yh, y):
        pzero = sigmoid(par['lpzero'])
        r = np.exp(-par['lalpha'])
        clike = np.clip(like0(r, yh, y), a_max=clip_like)
        like = pzero*(y==0) + (1-pzero)*np.exp(clike)
        return log(like)

    # pass to glm
    par = glm(y, x=x, fe=fe, data=data, extra=extra, link=link, loss=loss, **kwargs)

    # transform params
    par['pzero'] = sigmoid(par.pop('lpzero'))
    par['alpha'] = np.exp(-par.pop('lalpha'))

    return par

# ordinary least squares (just for kicks)
def ordinary_least_squares(y, x=[], fe=[], data=None, **kwargs):
    # base poisson distribution
    link = links['identity']
    loss0 = losses['least_squares']
    extra = {'lsigma': 0.0}

    # zero inflation
    def loss(par, yh, y):
        lsigma = par['lsigma']
        sigma2 = np.exp(2*lsigma)
        like = -lsigma + 0.5*loss0(yh, y)/sigma2
        return like

    # pass to glm
    par, sig = glm(y, x=x, fe=fe, data=data, extra=extra, link=link, loss=loss, **kwargs)

    # transform params
    par['sigma'] = np.exp(par.pop('lsigma'))

    return par, sig
