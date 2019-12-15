from jax import jit, grad, jacobian
import jax.numpy as np
import jax.random as random
import numpy as np0
import scipy.sparse as sp

from .design import design_matrices
from .summary import param_table

##
## constants
##

eps = 1e-7

##
## batching it
##

class DataLoader:
    def __init__(self, y, x, batch_size):
        self.y = y
        self.x = x
        self.batch_size = batch_size
        self.data_size = len(y)
        self.num_batches = self.data_size // batch_size
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

# maximum likelihood using jax -  this expects a mean log likelihood
# can only handle dense x
def maxlike(y, x, model, params0, batch_size=4092, epochs=3, learning_rate=0.5, output=None):
    # compute derivatives
    g0_fun = grad(model)
    h0_fun = jacobian(g0_fun)

    # generate functions
    f_fun = jit(model)
    g_fun = jit(g0_fun)
    h_fun = jit(h0_fun)

    # construct dataset
    N, K = len(y), len(params0)
    data = DataLoader(y, x, batch_size)

    # initialize params
    params = params0.copy()

    # do training
    for ep in range(epochs):
        # epoch stats
        agg_loss, agg_batch = 0.0, 0

        # iterate over batches
        for y_bat, x_bat in data:
            # compute gradients
            loss = f_fun(params, y_bat, x_bat)
            diff = g_fun(params, y_bat, x_bat)

            # compute step
            step = learning_rate*diff
            params -= step

            # error
            gain = np.dot(step, diff)
            move = np.max(np.abs(gain))

            # compute statistics
            agg_loss += loss
            agg_batch += 1

        # display stats
        avg_loss = agg_loss/agg_batch
        print(f'{ep:3}: loss = {avg_loss}')

    # return to device
    if output == 'beta':
        return params.copy()

    # get hessian matrix
    hess = np.zeros((K, K))
    for y_bat, x_bat in data:
        hess += h_fun(params, y_bat, x_bat)
    hess *= batch_size/N

    # get cov matrix
    sigma = np.linalg.inv(hess)/N

    # return all
    return params.copy(), sigma.copy()

# link functions
links = {
    'identity': lambda x: x,
    'exponential': lambda x: np.exp(x),
}

# loss functions
losses = {
    'least_squares': lambda yh, y: (y-yh)**2,
    'poisson': lambda yh, y: yh - y*np.log(yh+eps)
}

# default glm specification
def glm(y, x=[], fe=[], data=None, extra=[], link=None, loss=None, intercept=True, drop='first', output=None, table=True, **kwargs):
    # construct design matrices
    y_vec, x_mat, x_names = design_matrices(y, x=x, fe=fe, data=data, intercept=intercept, drop=drop)
    N, K = x_mat.shape

    # account for extra params
    P = len(extra)
    x_names = extra + x_names

    # evaluator
    def model(par, y, x):
        linear = np.dot(x, par[-K:])
        pred = link(linear)
        like = loss(par, pred, y)
        return np.mean(like)

    # estimate model
    params = np.zeros(P+K)
    beta, sigma = maxlike(y_vec, x_mat, model, params, output=output, **kwargs)

    # return relevant
    if table:
        return param_table(beta, sigma, x_names)
    else:
        return beta, sigma

# poisson regression
def poisson(y, x=[], fe=[], data=None, **kwargs):
    link = links['exponential']
    loss = lesses['poisson']
    return glm(y, x=x, fe=fe, data=data, link=link, loss=loss, **kwargs)

# zero inflated poisson regression
def zero_inflated_poisson(y, x=[], fe=[], data=None, **kwargs):
    # base poisson distribution
    link = links['exponential']
    loss0 = losses['poisson']
    extra = ['pzero']

    # zero inflation
    def loss(par, yh, y):
        pzero = 1/(1+np.exp(-par[0]))
        like = pzero*(y==0) + (1-pzero)*np.exp(-loss0(yh, y))
        return -np.log(like)

    return glm(y, x=x, fe=fe, data=data, extra=extra, link=link, loss=loss, **kwargs)

# ordinary least squares
def ordinary_least_squares(y, x=[], fe=[], data=None, **kwargs):
    # base poisson distribution
    link = links['identity']
    loss0 = losses['least_squares']
    extra = ['sigma']

    # zero inflation
    def loss(par, yh, y):
        lsigma = par[0]
        sigma2 = np.exp(2*lsigma)
        like = lsigma + 0.5*loss0(yh, y)/sigma2
        return like

    return glm(y, x=x, fe=fe, data=data, extra=extra, link=link, loss=loss, **kwargs)
