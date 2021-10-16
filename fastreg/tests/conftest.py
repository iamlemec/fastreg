import pytest
import fastreg as fr

@pytest.fixture(params=[10_000, 1_000_000])
def data(request):
    N = request.param
    return fr.dataset(N=N, K1=10, K2=100, models=['linear', 'poisson'])
