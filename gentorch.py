import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import grad, Variable

from .design import design_matrices
from .summary import param_table

##
## constants
##

eps = 1e-7

##
## sparse
##

# make a sparse tensor
def sparse_tensor(inp):
    mat = inp.tocoo()
    idx = torch.LongTensor(np.vstack([mat.row, mat.col]))
    val = torch.FloatTensor(mat.data)
    return torch.sparse.FloatTensor(idx, val, mat.shape)

##
## derivates
##

def flatgrad(y, x, **kwargs):
    return torch.flatten(grad(y, x, **kwargs)[0])

def vecgrad(y, x, **kwargs):
    units = torch.eye(y.numel(), device=y.device)
    rows = [flatgrad(y, x, grad_outputs=u, retain_graph=True) for u in units]
    return torch.stack(rows)

# looping for hessians
def hessian(y, xs):
    rows = []
    for xi in xs:
        dyi = flatgrad(y, xi, create_graph=True)
        cols = [vecgrad(dyi, xj) for xj in xs]
        rows.append(torch.cat(cols, 1))
    return torch.cat(rows, 0)

##
## estimation
##

# maximum likelihood using torch -  this expects a mean log likelihood
# can only handle dense x (sparse hessian balks)
def maxlike(y, x, model, params, batch_size=4092, epochs=3, learning_rate=0.5, dtype=np.float32, device='cpu', output=None):
    # get data size
    N = len(y)

    # convert to tensors
    y_ten = torch.tensor(y, dtype=dtype, device=device)
    x_ten = torch.tensor(x, dtype=dtype, device=device)

    dset = TensorDataset(x_ten, y_ten)
    dlod = DataLoader(dset, batch_size)

    # create optimizer
    optim = torch.optim.SGD(params, lr=learning_rate)

    # do training
    for ep in range(epochs):
        # epoch stats
        agg_loss, agg_batch = 0.0, 0

        # iterate over batches
        for x_bat, y_bat in dlod:
            # compute gradients
            loss = model(y_bat, x_bat)
            loss.backward()

            # implement update
            optim.step()
            optim.zero_grad()

            # compute statistics
            agg_loss += loss
            agg_batch += 1

        # display stats
        avg_loss = agg_loss/agg_batch
        print(f'{ep:3}: loss = {avg_loss}')

    # construct params (flatified)
    beta = torch.cat([p.flatten() for p in params]).detach().cpu().numpy()

    # just params
    if output == 'beta':
        return beta

    # get hessian (flatified) - but what about off diagonal terms?
    K = sum([p.numel() for p in params])
    fisher = np.zeros((K, K))
    for x_bat, y_bat in dlod:
        loss = model(y_bat, x_bat)
        hess = hessian(loss, params)
        fisher += hess.detach().cpu().numpy()
    fisher *= batch_size/N

    # get cov matrix
    sigma = np.linalg.inv(fisher)/N

    # return all
    return beta, sigma

# default glm specification
link0 = lambda x: x
loss0 = lambda i, o: torch.pow(i-o, 2)

def glm(y, x=[], fe=[], data=None, link=link0, loss=loss0, params=[], intercept=True, drop='first', dtype=torch.float, device='cpu', output=None, **kwargs):
    if len(x) == 0 and len(fe) == 0 and not intercept:
        raise(Exception('No columns present!'))

    # construct design matrices
    y_vec, x_mat, x_names = design_matrices(y, x=x, fe=fe, data=data, intercept=intercept, drop=drop, output='dense')
    N, K = x_mat.shape

    # linear layer
    linear = torch.nn.Linear(K, 1, bias=False).to(dtype=dtype, device=device)

    # collect params
    params1 = [linear.weight] + params

    # evaluator
    def model(y, x):
        inter = torch.flatten(linear(x))
        pred = link(inter)
        like = loss(pred, y)
        return torch.mean(like)

    # estimate model
    beta, sigma = maxlike(y_vec, x_mat, model, params1, dtype=dtype, device=device, **kwargs)

    # extract linear layer
    table = param_table(beta[:K], sigma[:K, :K], x_names)

    # return relevant
    if output == 'table':
        return table
    else:
        return table, beta, sigma

# logit regression
def logit(y, x=[], fe=[], data=None, **kwargs):
    link = lambda x: torch.exp(x)
    loss = lambda yh, y: torch.log(1+yh) - y*torch.log(yh+eps)
    return glm(y, x=x, fe=fe, data=data, link=link, loss=loss, **kwargs)

# poisson regression
def poisson(y, x=[], fe=[], data=None, **kwargs):
    link = lambda x: torch.exp(x)
    loss = lambda yh, y: yh - y*torch.log(yh+eps)
    return glm(y, x=x, fe=fe, data=data, link=link, loss=loss, **kwargs)

def zero_inflated_poisson(y, x=[], fe=[], data=None, **kwargs):
    # zero probability
    spzero = Variable(-2*torch.ones(1), requires_grad=True)

    # base poisson distribution
    link = lambda x: torch.exp(x)
    loss0 = lambda yh, y: yh - y*torch.log(yh+eps)

    # zero inflation
    def loss(yh, y):
        pzero = torch.sigmoid(spzero)
        like = pzero*(y==0) + (1-pzero)*torch.exp(-loss0(yh, y))
        return -torch.log(like)

    return glm(y, x=x, fe=fe, data=data, link=link, loss=loss, params=[spzero], **kwargs)

# def negative_binomial(y, x, data, **kwargs):
# def zero_inflated_negative_binomial(y, x, data, **kwargs):
