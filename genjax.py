from jax import jit, grad, jacobian
import jax.numpy as np
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
def maxlike(y, x, model, params0, batch_size=4092, epochs=3, learning_rate=0.5):
    # get data size
    N = len(y)
    K = len(params0)

    # construct dataset
    dlod = DataLoader(y, x, batch_size)

    # compute derivatives
    g0_fun = grad(model)
    h0_fun = jacobian(g0_fun)

    # generate gradient function
    f_fun = jit(model)
    g_fun = jit(g0_fun)
    h_fun = jit(h0_fun)

    # initialize params
    params = params0.copy()

    # do training
    for ep in range(epochs):
        # epoch stats
        agg_loss, agg_batch = 0.0, 0

        # iterate over batches
        for y_bat, x_bat in dlod:
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

    # get hessian (flatified) - but what about off diagonal terms?
    fisher = np0.zeros((K, K))
    for y_bat, x_bat in dlod:
        fisher += h_fun(params, y_bat, x_bat)
    fisher *= batch_size/N

    # get cov matrix
    sigma = np0.linalg.inv(fisher)/N

    # return all
    return params, sigma

# default glm specification
link0 = lambda x: x
loss0 = lambda i, o: (i-o)**2
def glm(y, x=[], fe=[], data=None, link=link0, loss=loss0, intercept=True, drop='first', output='table', **kwargs):
    if len(x) == 0 and len(fe) == 0 and not intercept:
        raise(Exception('No columns present!'))

    # construct design matrices
    y_vec, x_mat, x_names = design_matrices(y, x=x, fe=fe, data=data, intercept=intercept, drop=drop)
    N, K = x_mat.shape

    # evaluator
    def model(par, y, x):
        linear = np.dot(x, par)
        pred = link(linear)
        like = loss(pred, y)
        return np.mean(like)

    # estimate model
    params = np.zeros(K)
    beta, sigma = maxlike(y_vec, x_mat, model, params, **kwargs)

    # return relevant
    if output == 'table':
        return param_table(beta, sigma, x_names)
    else:
        return beta, sigma

# poisson regression
def poisson(y, x=[], fe=[], data=None, **kwargs):
    link = lambda x: np.exp(x)
    loss = lambda yh, y: yh - y*np.log(yh+eps)
    return glm(y, x=x, fe=fe, data=data, link=link, loss=loss, **kwargs)
