import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats.distributions import norm
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import grad

from .design import design_matrices

# constants
eps = 1e-7

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

# generic GLM model
class GlmModel(torch.nn.Module):
    def __init__(self, K, link):
        super().__init__()
        self.linear = torch.nn.Linear(K, 1, bias=False)
        self.link = link

    def forward(self, x):
        x = torch.flatten(self.linear(x))
        x = self.link(x)
        return x

# default specification
link0 = lambda x: x
loss0 = lambda i, o: torch.pow(i-o, 2)

# glm regression using torch
def glm(y, x=[], fe=[], data=None, intercept=True, drop='first', link_fn=link0, loss_fn=loss0, batch_size=4092, epochs=3, learning_rate=0.5):
    if len(x) == 0 and len(fe) == 0 and not intercept:
        raise(Exception('No columns present!'))

    # choose best device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # construct design matrices
    y_vec, x_mat, x_names = design_matrices(y, x, fe, data, intercept=intercept, drop=drop)
    N, K = x_mat.shape

    # convert to tensors
    y_ten = torch.from_numpy(y_vec.astype(np.float32)).to(device)
    x_ten = torch.from_numpy(x_mat.astype(np.float32)).to(device)

    # create dataset and dataloader
    dset = TensorDataset(x_ten, y_ten)
    dlod = DataLoader(dset, batch_size, shuffle=True)

    # construct model
    model = GlmModel(K, link_fn).to(device)

    # create optimizer
    params = [model.linear.weight]
    optim = torch.optim.SGD(params, lr=learning_rate)

    # do training
    for ep in range(epochs):
        # epoch stats
        agg_loss, agg_batch = 0.0, 0

        # iterate over batches
        for x_bat, y_bat in dlod:
            # compute gradients
            pred = model(x_bat)
            lvec = loss_fn(pred, y_bat)
            loss = torch.mean(lvec)
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

    # construct params
    betas = model.linear.weight.detach().cpu().numpy().flatten()
    table = pd.DataFrame({'coeff': betas}, index=x_names)

    # get hessian
    fisher = np.zeros((K, K))
    for x_bat, y_bat in dlod:
        pred = model(x_bat)
        lvec = loss_fn(pred, y_bat)
        loss = torch.mean(lvec)
        hess = hessian(loss, model.linear.weight, device=device)
        fisher += hess.detach().cpu().numpy()
    fisher *= batch_size/N

    # get cov matrix
    cov = np.linalg.inv(fisher)/N
    stderr = np.sqrt(cov.diagonal())

    # confidence interval
    s95 = norm.ppf(0.975)
    low95 = betas - s95*stderr
    high95 = betas + s95*stderr

    # p-value
    zscore = betas/stderr
    pvalue = 1 - norm.cdf(np.abs(zscore))

    # store for return
    table['stderr'] = stderr
    table['low95'] = low95
    table['high95'] = high95
    table['pvalue'] = pvalue

    # return all
    return table

# standard poisson regression
def poisson(y, x=[], fe=[], data=None, **kwargs):
    link = lambda x: torch.exp(x)
    loss = lambda yh, y: yh - y*torch.log(yh+eps)
    return glm(y, x=x, fe=fe, data=data, link_fn=link, loss_fn=loss, **kwargs)
