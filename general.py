##
## tensorflow tools
##

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
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", shape=[self.vocabulary_size, self.num_units]
        )
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.num_units])

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
    None: lambda x: np.ones_like(x),
    'exp': lambda x: x
}

# default loss derivatives (-log(likelihood))
eps = 1e-7
dloss_default = {
    'mse': lambda y, yh: yh - y,
    'poisson': lambda y, yh: ( yh - y ) / ( yh + eps )
}

# glm regression using keras
def glm(y, x=[], fe=[], data=None, intercept=True, drop='first', output='params', link='identity', loss='mse', batch_size=4092, epochs=3, learning_rate=0.5, metrics=['accuracy'], dlink=None, dloss=None):
    if type(link) in (None, str):
        dlink = dlink_default.get(link, None)
    if type(loss) is str:
        dloss = dloss_default.get(loss, None)

    if type(link) is str:
        if link == 'identity':
            link = tf.identity
        else:
            link = getattr(K, link)
    if type(loss) is str:
        loss = getattr(keras.losses, loss)

    # construct design matrices
    y_vec, x_mat, fe_mat, x_names, fe_names = design_matrices(
        y, x, fe, data, intercept=intercept, drop=drop, separate=True
    )
    N = len(y_vec)

    # collect model components
    x_data = [] # actual data
    inputs = [] # input placeholders
    linear = [] # linear layers
    inter = [] # intermediate values
    names = [] # coefficient names

    # check dense factors
    if x_mat is not None:
        _, Kd = x_mat.shape
        inputs_dense = layers.Input((Kd,))
        linear_dense = layers.Dense(1, use_bias=False)
        inter_dense = linear_dense(inputs_dense)

        x_data.append(x_mat)
        inputs.append(inputs_dense)
        linear.append(linear_dense)
        inter.append(inter_dense)
        names.append(x_names)

    # check sparse factors
    if fe_mat is not None:
        _, Ks = fe_mat.shape
        fe_ten = sparse_tensor(fe_mat)
        inputs_sparse = layers.Input((Ks,), sparse=True)
        linear_sparse = SparseLayer(Ks, 1, use_bias=False)
        inter_sparse = linear_sparse(inputs_sparse)

        x_data.append(fe_ten)
        inputs.append(inputs_sparse)
        linear.append(linear_sparse)
        inter.append(inter_sparse)
        names.append(fe_names)

    # construct network
    core = layers.Add()(inter) if len(inter) > 1 else inter[0]
    pred = keras.layers.Lambda(link)(core)
    model = keras.Model(inputs=inputs, outputs=pred)

    # run estimation
    optim = keras.optimizers.Adagrad(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    model.fit(x_data, y_vec, epochs=epochs, batch_size=batch_size)

    # construct params
    names = sum(names, [])
    betas = tf.concat([tf.reshape(act.weights[0], (-1,)) for act in linear], 0)
    coeff = pd.Series(dict(zip(names, betas.numpy())))

    # generate predictions
    # y_hat = model.predict(x_data).flatten()

    # calculate standard errors
    # if dlink is not None and dloss is not None:
    #     dlink_vec = dlink(y_hat)
    #     dloss_vec = dloss(y_vec, y_hat)
    #     dpred_vec = dlink_vec*dloss_vec
    #     dlike_dense = dpred_vec[:,None]*x_mat
    #     fisher_dense = np.matmul(dlike_dense.T, dlike_dense)
    #     dlike_sparse = fe_mat.multiply(dpred_vec[:, None])
    #     fisher_sparse = (dlike_sparse.T*dlike_sparse)

    # return
    if output == 'params':
        return coeff
    elif output == 'model':
        return model, coeff

# standard poisson regression
def poisson(y, x=[], fe=[], data=None, **kwargs):
    return glm(y, x=x, fe=fe, data=data, link='exp', loss='poisson', **kwargs)

# try negative binomial next (custom loss)
