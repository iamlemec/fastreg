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

# standard poisson regression
def poisson(y, x=[], fe=[], data=None, backend='keras', **kwargs):
    return glm(y, x=x, fe=fe, data=data, link='exp', loss='poisson', **kwargs)
