import jax
import jax.lax as lax
import jax.scipy.special as spec
import jax.numpy as np
from jax.tree_util import tree_flatten, tree_map, tree_leaves
import numpy as np0
import scipy.sparse as sp

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
## batching it
##

class DataLoader:
    def __init__(self, y, x, batch_size):
        self.y = y
        self.x = x
        self.batch_size = batch_size
        self.data_size = len(y)
        self.num_batches = max(1, self.data_size // batch_size)
        self.sparse = sp.issparse(x)

    def __iter__(self):
        loc = 0
        for i in range(self.num_batches):
            by, bx = self.y[loc:loc+self.batch_size], self.x[loc:loc+self.batch_size, :]
            if self.sparse:
                bx = bx.toarray()
            yield by, bx
            loc += self.batch_size

##
## estimation
##

# maximum likelihood using jax - this expects a mean log likelihood
def maxlike(y, x, model, params, batch_size=4092, epochs=3, learning_rate=0.5, eta=0.001, gamma=0.9, per=100, disp=None, stderr=True):
    # compute derivatives
    vg_fun = jax.jit(jax.value_and_grad(model))
    g_fun = jax.jit(jax.grad(model))
    h_fun = jax.jit(jax.hessian(model))

    # construct dataset
    N = len(y)
    data = DataLoader(y, x, batch_size)

    # initialize params
    grms = {k: np.zeros_like(v) for k, v in params.items()}

    # do training
    for ep in range(epochs):
        # epoch stats
        agg_loss, agg_batch = 0.0, 0

        # iterate over batches
        for b, (y_bat, x_bat) in enumerate(data):
            # compute gradients
            loss, grad = vg_fun(params, y_bat, x_bat)

            lnan = np.isnan(loss)
            gnan = tree_map(lambda g: np.isnan(g).any(), grad)

            if lnan or np.any(tree_leaves(gnan)):
                print('Encountered nans!')
                return params, None

            for k in params:
                grms[k] += (1-gamma)*(grad[k]**2-grms[k])
                params[k] += eta*grad[k]/np.sqrt(grms[k]+eps)

            # compute statistics
            agg_loss += loss
            agg_batch += 1

        # display stats
        avg_loss = agg_loss/agg_batch

        # display output
        if disp is not None:
            disp(ep, avg_loss, params)

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
        sigma = {k1: {k2: -np.linalg.inv(v2)/N for k2, v2 in v1.items()} for k1, v1 in hess.items()}
    elif stderr is None:
        sigma = N*hess
    else:
        # use block diagonal inverse formula
        # https://en.wikipedia.org/wiki/Invertible_matrix#Blockwise_inversion
        pass

    # return to device
    return params, sigma

# default glm specification
def glm(y, x=[], fe=[], data=None, extra={}, link=None, loss=None, intercept=True, drop='first', stderr=True, **kwargs):
    # construct design matrices
    y_vec, x_mat, x_names = design_matrices(y, x=x, fe=fe, data=data, intercept=intercept, drop=drop)
    N, K = x_mat.shape

    # evaluator
    def model(par, ydat, xdat):
        beta = par['beta']
        linear = xdat @ beta
        pred = link(linear)
        like = loss(par, pred, ydat)
        return np.mean(like)

    # displayer
    def disp(e, l, p):
        beta = p['beta']
        print(f'[{e:3d}] {l:.4f}: {np.mean(beta):.4f} {np.std(beta):.4f}')

    # estimate model
    params0 = {'beta': np.zeros(K), **extra}
    params, sigma = maxlike(y_vec, x_mat, model, params0, stderr=stderr, disp=disp, **kwargs)

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
    par = glm(y, x=x, fe=fe, data=data, extra=extra, link=link, loss=loss, **kwargs)

    # transform params
    par['pzero'] = sigmoid(par.pop('lpzero'))

    return par

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
    par = glm(y, x=x, fe=fe, data=data, extra=extra, link=link, loss=loss, **kwargs)

    # transform params
    par['sigma'] = np.exp(par.pop('lsigma'))

    return par
