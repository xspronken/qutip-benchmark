"""This file contains the benchmarks that are run the benchmark.py script."""
import pytest
import qutip
import scipy
import numpy as np


# Available datatypes (using qutip_dense and qutip_csr to avoid confusion
# with density parameters)
@pytest.fixture(params=['numpy', 'scipy_csr', 'qutip_dense', 'qutip_csr'])
def dtype(request): return request.param


@pytest.fixture(params=np.logspace(1, 9, 9, base=2, dtype=int).tolist())
def size(request): return request.param


@pytest.fixture(params=["dense", "sparse"])
def density(request): return request.param


@pytest.fixture()
def left_oper(size, density, dtype):
    """Return a random matrix of size `sizexsize'. Density is either 'dense'
    or 'sparse' and returns a fully dense or a tridiagonal matrix respectively.
    The matrices are Hermitian."""
    np.random.seed(1)

    if density == "sparse":
        ofdiag = np.random.rand(size-1) + 1j*np.random.rand(size-1)
        diag = np.random.rand(size) + 1j*np.random.rand(size)

        res = (np.diag(ofdiag, k=-1) +
               np.diag(diag, k=0) +
               np.diag(ofdiag.conj(), k=1))

    elif density == "dense":
        H = np.random.random((size, size)) + 1j*np.random.random((size, size))
        res = H + H.T.conj()

    if dtype == 'numpy':
        return res
    elif dtype == 'scipy_csr':
        return scipy.sparse.csr_matrix(res)
    else:
        res = qutip.Qobj(res)
        # the to() method only accepts dense or csr as inputs
        return res.to(dtype[6:])


@pytest.fixture()
def right_oper(size, density, dtype):
    """Return a random matrix of size `sizexsize'. Density is either 'dense'
    or 'sparse' and returns a fully dense or a tridiagonal matrix respectively.
    The matrices are Hermitian."""
    np.random.seed(1)

    if density == "sparse":
        ofdiag = np.random.rand(size-1) + 1j*np.random.rand(size-1)
        diag = np.random.rand(size) + 1j*np.random.rand(size)

        res = (np.diag(ofdiag, k=-1) +
               np.diag(diag, k=0) +
               np.diag(ofdiag.conj(), k=1))

    elif density == "dense":
        H = np.random.random((size, size)) + 1j*np.random.random((size, size))
        res = H + H.T.conj()

    if dtype == 'numpy':
        return res
    if dtype == 'scipy_csr':
        return scipy.sparse.csr_matrix(res)
    res = qutip.Qobj(res)
    # the to() method only accepts dense or csr as inputs
    return res.to(dtype[6:])


@pytest.fixture()
def right_ket(size, density, dtype):
    if density == "sparse":
        res = qutip.rand_ket(size, density=0.3)
    else:
        res = qutip.rand_ket(size, density=1)

    if dtype == 'numpy':
        return res.full()
    if dtype == 'scipy_csr':
        return scipy.sparse.csr_matrix(res.full())
    return res.to(dtype[6:])


def add(left, right):
    return left+right


def matmul(left, right):
    return left@right


def test_add(benchmark, left_oper, right_oper, request):
    # Group benchmark by operation, density and size.
    group = request.node.callspec.id
    group = "Add-" + group
    benchmark.group = group

    left = left_oper
    right = right_oper

    # Benchmark operations
    result = benchmark(add, left, right)

    return result


def test_matmul_oper_oper(benchmark, left_oper, right_oper, request):
    # Group benchmark by operation, density and size.
    group = request.node.callspec.id
    group = "Matmul_op@op-" + group
    benchmark.group = group

    left = left_oper
    right = right_oper

    # Benchmark operations
    result = benchmark(matmul, left, right)

    return result


def test_matmul_oper_ket(benchmark, left_oper, right_ket, request):
    # Group benchmark by operation, density and size.
    group = request.node.callspec.id
    group = "Matmul_op@ket-" + group
    benchmark.group = group

    left = left_oper
    right = right_ket

    # Benchmark operations
    result = benchmark(matmul, left, right)

    return result
