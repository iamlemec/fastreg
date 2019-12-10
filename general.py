import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats.distributions import norm
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import grad

from .design import design_matrices

##
## constants
##

eps = 1e-7

##
## tools
##

def param_table(beta, sigma, names):
    # standard errors
    stderr = np.sqrt(sigma.diagonal())

    # confidence interval
    s95 = norm.ppf(0.975)
    low95 = beta - s95*stderr
    high95 = beta + s95*stderr

    # p-value
    zscore = beta/stderr
    pvalue = 1 - norm.cdf(np.abs(zscore))

    # return all
    return pd.DataFrame({
        'coeff': beta,
        'stderr': stderr,
        'low95': low95,
        'high95': high95,
        'pvalue': pvalue
    }, index=names)

##
## glm with torch
##

def flatgrad(y, x, **kwargs):
    return torch.flatten(grad(y, x, **kwargs)[0])

# looping for hessians
def hessian(y, x, device=None):
    dy = flatgrad(y, x, create_graph=True)
    units = torch.eye(dy.numel(), device=device)
    rows = [flatgrad(dy, x, grad_outputs=u, retain_graph=True) for u in units]
    return torch.stack(rows)

# maximum likelihood using torch
def maxlike(y, x, model, params, batch_size=4092, epochs=3, learning_rate=0.5, dtype=np.float32, device='cpu'):
    # get data size
    N = len(y)

    # convert to tensors
    y_ten = torch.from_numpy(y.astype(dtype)).to(device)
    x_ten = torch.from_numpy(x.astype(dtype)).to(device)

    # create dataset and dataloader
    dset = TensorDataset(x_ten, y_ten)
    dlod = DataLoader(dset, batch_size, shuffle=True)

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
    beta = [p.detach().cpu().numpy().flatten() for p in params]

    # get hessian (flatified)
    K = [p.numel() for p in params]
    fisher = [np.zeros((s, s)) for s in K]
    for x_bat, y_bat in dlod:
        loss = model(y_bat, x_bat)
        for i, p in enumerate(params):
            hess = hessian(loss, p, device=device)
            fisher[i] += hess.detach().cpu().numpy()
    for i in range(len(params)):
        fisher[i] *= batch_size/N

    # get cov matrix
    sigma = [np.linalg.inv(f)/N for f in fisher]

    # return all
    return beta, sigma

# default specification
link0 = lambda x: x
loss0 = lambda i, o: torch.pow(i-o, 2)
def glm(y, x, data, link=link0, loss=loss0, params=[], intercept=True, drop='first', device='cpu', output='single', **kwargs):
    if len(x) == 0 and len(fe) == 0 and not intercept:
        raise(Exception('No columns present!'))

    # construct design matrices
    y_vec, x_mat, x_names = design_matrices(y, x, data=data, intercept=intercept, drop=drop)
    N, K = x_mat.shape

    # linear layer
    linear = torch.nn.Linear(K, 1, bias=False).to(device)

    # collect params
    params = [linear.weight] + params

    # evaluator
    def model(y, x):
        inter = torch.flatten(linear(x))
        pred = link(inter)
        like = loss(pred, y)
        return torch.mean(like)

    # estimate model
    beta, sigma = maxlike(y_vec, x_mat, model, params, device=device, **kwargs)

    # interpret results
    if output == 'full':
        return beta, sigma
    elif output == 'single':
        return param_table(beta[0], sigma[0], x_names)

# poisson regression
def poisson(y, x, data, **kwargs):
    link = lambda x: torch.exp(x)
    loss = lambda yh, y: yh - y*torch.log(yh+eps)
    return glm(y, x, data, link=link, loss=loss, **kwargs)

# logit regression
def logit(y, x, data, **kwargs):
    link = lambda x: torch.exp(x)
    loss = lambda yh, y: torch.log(1+yh) - y*torch.log(yh+eps)
    return glm(y, x, data, link=link, loss=loss, **kwargs)
