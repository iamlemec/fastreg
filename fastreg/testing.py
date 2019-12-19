import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# true parameters
c = {
    'one': 0.0,
    'sigma': 1.0,
    'x1': 0.6,
    'x2': 0.2,
    'id1': 0.2,
    'id2': 0.5,
    'pz': 0.2,
    'alpha': 0.3
}

# poisson dampening
pfact = 100

# good negbin in terms of mean and overdispersion (var = m + alpha*m^2)
def rand_negbin(mean, alpha, size=None, state=np.random):
    return state.negative_binomial(1/alpha, 1/(1+alpha*mean), size=size)

def dataset(N=1_000_000, K1=10, K2=100, seed=89320432):
    # init random
    st = np.random.RandomState(seed)

    # core regressors
    df = pd.DataFrame({
        'id1': st.randint(K1, size=N),
        'id2': st.randint(K2, size=N),
        'x1': st.randn(N),
        'x2': st.randn(N)
    })

    # predictors
    df['yhat0'] = c['one'] + c['x1']*df['x1'] + c['x2']*df['x2']
    df['yhat'] = df['yhat0'] + c['id1']*df['id1']/pfact + c['id2']*df['id2']/pfact

    # linear
    df['y0'] = df['yhat0'] + c['sigma']*st.randn(N)
    df['y'] = df['yhat'] + c['sigma']*st.randn(N)

    # logit
    df['Eb0'] = 1/(1+np.exp(-df['yhat0']))
    df['Eb'] = 1/(1+np.exp(-df['yhat']))
    df['b0'] = (st.randn(N) < df['Eb0']).astype(np.int)
    df['b'] = (st.randn(N) < df['Eb']).astype(np.int)

    # poisson
    df['Ep0'] = np.exp(df['yhat0'])
    df['Ep'] = np.exp(df['yhat'])
    df['p0'] = st.poisson(df['Ep0'])
    df['p'] = st.poisson(df['Ep'])

    # zero-inflated poisson
    df['pz0'] = np.where(st.rand(N) < c['pz'], 0, df['p0'])
    df['pz'] = np.where(st.rand(N) < c['pz'], 0, df['p'])

    # negative binomial
    df['nb0'] = rand_negbin(df['Ep0'], c['alpha'], state=st)
    df['nb'] = rand_negbin(df['Ep'], c['alpha'], state=st)

    # zero-inflated poisson
    df['nbz0'] = np.where(st.rand(N) < c['pz'], 0, df['nb0'])
    df['nbz'] = np.where(st.rand(N) < c['pz'], 0, df['nb'])

    return df

def plot_coeff(beta):
    coeff = pd.DataFrame({
        'id2': np.arange(len(beta)),
        'beta1': beta
    })
    coeff['beta0'] = c['id2']*coeff['id2']/pfact
    coeff['beta0'] -= coeff['beta0'].mean()
    coeff['beta1'] -= coeff['beta1'].mean()

    # inferred ranges
    bmin = coeff[['beta0', 'beta1']].min().min()
    bmax = coeff[['beta0', 'beta1']].max().max()
    bvec = np.linspace(bmin, bmax, 100)

    # plot estimates
    fig, ax = plt.subplots(figsize=(6, 5))
    coeff.plot.scatter(x='beta0', y='beta1', ax=ax, alpha=0.5);
    ax.plot(bvec, bvec, c='r', linewidth=1, zorder=1);
    fig.show()

def test_ols(data, y='y', x=['x1', 'x2'], fe=['id1', 'id2'], plot=False, **kwargs):
    from . import linear

    table = linear.ols(y=y, x=x, fe=fe, data=data, **kwargs)

    if plot:
        plot_coeff(table['coeff'].filter(regex='id2'))

    return table

def test_jax(data, estim='poisson', y='p', x=['x1', 'x2'], fe=['id1', 'id2'], plot=False, **kwargs):
    from . import general

    if type(estim) is str:
        estim = getattr(general, estim)

    table = estim(y=y, x=x, fe=fe, data=data, **kwargs)

    if plot:
        plot_coeff(table['coeff'].filter(regex='id2'))

    return table

def test_torch(data, estim='poisson', y='p', x=['x1', 'x2'], fe=['id1', 'id2'], plot=False, **kwargs):
    from . import gentorch

    if type(estim) is str:
        estim = getattr(gentorch, estim)

    table = estim(y=y, x=x, fe=fe, data=data, **kwargs)

    if plot:
        plot_coeff(table['coeff'].filter(regex='id2'))

    return table
