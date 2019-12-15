import numpy as np
import pandas as pd
import fastreg.genjax as frx

import matplotlib.pyplot as plt

# true parameters
c = {
    '1': 0.0,
    'x1': 0.6,
    'x2': 0.2,
    'id1': 0.2,
    'id2': 0.5,
    'pz': 0.2
}

# poisson dampening
pfact = 100

def dataset(N=1_000_000, K1=10, K2=100, state=89320432):
    # init random
    st = np.random.RandomState(state)

    # core regressors
    df = pd.DataFrame({
        'one': 1,
        'id1': st.randint(K1, size=N),
        'id2': st.randint(K2, size=N),
        'x1': st.randn(N),
        'x2': st.randn(N)
    })

    # predictors
    df['yhat0'] = c['1'] + c['x1']*df['x1'] + c['x2']*df['x2']
    df['yhat'] = df['yhat0'] + c['id1']*df['id1']/pfact + c['id2']*df['id2']/pfact

    # ols outcomes
    df['y0'] = df['yhat0'] + st.randn(N)
    df['y'] = df['yhat'] + st.randn(N)

    # poisson outcomes
    df['Ep0'] = np.exp(df['yhat0'])
    df['Ep'] = np.exp(df['yhat'])
    df['p0'] = st.poisson(df['Ep0'])
    df['p'] = st.poisson(df['Ep'])
    df['pz'] = np.where(st.rand(N) < c['pz'], 0, df['p'])

    return df

def test_jax(data, estim='poisson', y='p', x=['x1', 'x2'], fe=['id1', 'id2'], plot=False, **kwargs):
    if type(estim) is str:
        estim = getattr(frx, estim)

    # run estimation
    table = estim(y=y, x=x, fe=fe, data=data, **kwargs)

    if plot:
        # extract estimates
        coeff = table['coeff'].filter(regex='id2').rename('beta1').rename_axis('id2').reset_index()
        coeff['id2'] = coeff['id2'].apply(lambda s: s[4:]).astype(np.int)
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

    return table
