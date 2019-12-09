import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats.distributions import norm
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from .design import design_matrices

##
## tensorflow tools
##

# general multiply
def multiply(x, y):
    if sp.issparse(x):
        return x.multiply(y)
    elif sp.issparse(y):
        return y.multiply(x)
    else:
        return x*y

# make a sparse tensor
def sparse_tensor(inp, dtype=np.float32):
    mat = inp.tocoo().astype(dtype)
    idx = list(zip(mat.row, mat.col))
    ten = tf.SparseTensor(idx, mat.data, mat.shape)
    return tf.sparse.reorder(ten)

# dense layer taking sparse matrix as input
class SparseLayer(layers.Layer):
    def __init__(self, vocabulary_size, num_units, activation=None, use_bias=True, **kwargs):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.num_units = num_units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', shape=[self.vocabulary_size, self.num_units])
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=[self.num_units])

    def call(self, inputs, **kwargs):
        is_sparse = isinstance(inputs, tf.SparseTensor)
        matmul = tf.sparse.sparse_dense_matmul if is_sparse else tf.matmul
        inters = matmul(inputs, self.kernel)
        outputs = tf.add(inters, self.bias) if self.use_bias else inters
        return self.activation(outputs)

    def compute_output_shape(self, input_shape):
        input_shape = input_shape.get_shape().as_list()
        return input_shape[0], self.num_units

##
## glm with keras
##

# default link derivatives
dlink_default = {
    'identity': lambda x: np.ones_like(x),
    'exp': lambda x: x
}

# default loss derivatives (-log(likelihood))
eps = 1e-7
dloss_default = {
    'mse': lambda y, yh: yh - y,
    'poisson': lambda y, yh: ( yh - y ) / ( yh + eps )
}

# glm regression using keras
def glm_keras(y, x=[], fe=[], data=None, intercept=True, drop='first', output='params', link='identity', loss='mse', batch_size=4092, epochs=3, learning_rate=0.5, valfrac=0.0, metrics=['accuracy'], dlink=None, dloss=None):
    if len(x) == 0 and len(fe) == 0 and not intercept:
        raise(Exception('No columns present!'))

    if type(link) is str:
        dlink = dlink_default.get(link, None)
    if type(loss) is str:
        dloss = dloss_default.get(loss, None)

    if type(link) is str:
        if link == 'identity':
            link = tf.identity
        else:
            link = getattr(keras.backend, link)
    if type(loss) is str:
        loss = getattr(keras.losses, loss)

    # construct design matrices
    y_vec, x_mat, x_names = design_matrices(y, x, fe, data, intercept=intercept, drop=drop)
    N, K = x_mat.shape

    # construct sparse tensor
    sparse = sp.issparse(x_mat)
    if sparse:
        x_ten = sparse_tensor(x_mat)
        linear = SparseLayer(K, 1, use_bias=False)
    else:
        x_ten = x_mat
        linear = layers.Dense(1, use_bias=False)

    # construct model
    inputs = layers.Input((K,), sparse=sparse)
    inter = linear(inputs)
    pred = keras.layers.Lambda(link)(inter)
    model = keras.Model(inputs=inputs, outputs=pred)

    # run estimation
    optim = keras.optimizers.Adagrad(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    model.fit(x_ten, y_vec, epochs=epochs, batch_size=batch_size, validation_split=valfrac)

    # construct params
    betas = linear.weights[0].numpy().flatten()
    table = pd.DataFrame({'coeff': betas}, index=x_names)

    # only point estimates
    if dlink is None or dloss is None:
        return table

    # compute link gradient
    y_hat = model.predict(x_ten, batch_size=batch_size).flatten()
    dlink_vec = dlink(y_hat)
    dloss_vec = dloss(y_vec, y_hat)
    dpred_vec = dlink_vec*dloss_vec

    # get fisher matrix
    dlike = multiply(x_mat, dpred_vec[:,None])
    fisher = dlike.T.dot(dlike)
    if sp.issparse(fisher):
        fisher = fisher.toarray()

    # get cov matrix
    cov = np.linalg.inv(fisher)
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

##
## torch
##

import torch
from torch.utils.data import TensorDataset, DataLoader

# generic GLM model
class GlmModel(torch.nn.Module):
    def __init__(self, K, link):
        super().__init__()
        self.linear = torch.nn.Linear(K, 1, bias=False)
        self.link = link

    def forward(self, x):
        x = self.linear(x)
        x = self.link(x)
        return x

# default specification
link0 = lambda x: x
loss0 = lambda i, o: torch.pow(i-o, 2)
dlink0 = lambda x: x
dloss0 = lambda yh, y: yh - y

# glm regression using torch
def glm_torch(y, x=[], fe=[], data=None, intercept=True, drop='first', output='params', link_fn=link0, loss_fn=loss0, dlink_fn=None, dloss_fn=None, batch_size=4092, epochs=3, learning_rate=0.5):
    if len(x) == 0 and len(fe) == 0 and not intercept:
        raise(Exception('No columns present!'))

    # construct design matrices
    y_vec, x_mat, x_names = design_matrices(y, x, fe, data, intercept=intercept, drop=drop)
    N, K = x_mat.shape

    # convert to tensors
    y_ten = torch.from_numpy(y_vec.astype(np.float32).reshape(-1, 1)).cuda()
    x_ten = torch.from_numpy(x_mat.astype(np.float32)).cuda()

    # create dataset and dataloader
    dset = TensorDataset(x_ten, y_ten)
    dlod = DataLoader(dset, batch_size, shuffle=True)

    # construct model
    model = GlmModel(K, link_fn).cuda()

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

    # return all
    return table

# standard poisson regression
def poisson_keras(y, x=[], fe=[], data=None, backend='keras', **kwargs):
    return glm_keras(y, x=x, fe=fe, data=data, link='exp', loss='poisson', **kwargs)

# standard poisson regression
def poisson_torch(y, x=[], fe=[], data=None, backend='keras', **kwargs):
    link = lambda x: torch.exp(x)
    loss = lambda yh, y: yh - y*torch.log(yh+eps)
    dlink = lambda x: torch.exp(x)
    dloss = lambda yh, y: (yh-y)/(yh+eps)
    return glm_torch(y, x=x, fe=fe, data=data, link_fn=link, loss_fn=loss, **kwargs)
