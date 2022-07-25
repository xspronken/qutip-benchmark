"""This file contains the benchmarks that are run the benchmark.py script."""

import pytest
import qutip as qt
import numpy as np




from . import benchmark_time_evo



time_evo_ops = [ getattr(benchmark_time_evo,_) for _ in dir(benchmark_time_evo) if _[:3]=="get"]
time_evo_ids = [ _[4:] for _ in dir(benchmark_time_evo) if _[:3]=="get"]


@pytest.fixture(params = np.logspace(1, 9, 9, base=2, dtype=int).tolist())
def size(request): return request.param

@pytest.fixture(params = ["dense", "sparse"])
def density(request): return request.param

@pytest.fixture(scope='function')
def matrix(size, density):
    """Return a random matrix of size `sizexsize'. Density is either 'dense'
    or 'sparse' and returns a fully dense or a tridiagonal matrix respectively.
    The matrices are Hermitian."""
    np.random.seed(1)


    if density == "sparse":
        ofdiag = np.random.rand(size-1) + 1j*np.random.rand(size-1)
        diag = np.random.rand(size) + 1j*np.random.rand(size)

        return qt.Qobj(np.diag(ofdiag, k=-1)
                + np.diag(diag, k=0)
                + np.diag(ofdiag.conj(), k=1))

    elif density == "dense":
        H = np.random.random((size, size)) + 1j*np.random.random((size, size))
        return qt.Qobj(H + H.T.conj())

@pytest.fixture(scope='function')
def vector(size,density):
    if density == "dense":
        return qt.rand_ket(size,density=1)
    return qt.rand_ket(size,density=0)

def time_dep(A, dtype):
    """Creates a Qobj evo with either string or function instanciation"""
    if dtype == 'function':
        def cos_t(t):
            return np.cos(t)
        return qt.QobjEvo([A,cos_t])
    elif dtype == 'string':
        return qt.QobjEvo([A,'cos(t)'])

#Supported dtypes
dtype = ['function','string']
@pytest.fixture(params = dtype)
def dtype(request): return request.param




@pytest.mark.parametrize("get_operation", time_evo_ops, ids=time_evo_ids)
def test_linear_algebra_time_evo(benchmark, matrix, vector, dtype, get_operation, request):
    # Group benchmark by operation, density and size.
    group = request.node.callspec.id
    group = group.split('-')
    benchmark.group = '-'.join(group)
    benchmark.extra_info['dtype'] = group[0]

    matrix = time_dep(matrix, dtype)

    # Benchmark operations and skip those that are not implemented.
    try:
        operation = get_operation()
        result = benchmark(operation, matrix, vector, 100)
    except (NotImplementedError):
        result = None

    return result
