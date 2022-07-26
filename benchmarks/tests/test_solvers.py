import time
import pytest
import numpy as np
import qutip as qt
from qutip.solver.options import SolverOptions
from . import benchmark_solver



rep = 2

solver_ops = [ getattr(benchmark_solver,_) for _ in dir(benchmark_solver) if _[:3]=="get"]
solver_ids = [ _[4:] for _ in dir(benchmark_solver) if _[:3]=="get"]



@pytest.fixture(params = np.logspace(2, 8, 7, base=2, dtype=int).tolist())
def dimension(request): return request.param

@pytest.fixture(params = ["jc","cavity","qubit"])
def model(request): return request.param




@pytest.fixture(scope='function')
def jc_setup(dimension):
    dimension = int(dimension/2)
    wa = 1
    wc = 1
    g = 2
    kappa = 0.5
    gamma = 0.1
    n_th = 0.75
    tspan = np.linspace(0, 10, 11)

    Ia = qt.qeye(2)
    Ic = qt.qeye(dimension)

    a = qt.destroy(dimension)
    at = qt.create(dimension)
    n = at * a

    sm = qt.sigmam()
    sp = qt.sigmap()
    sz = qt.sigmaz()

    H = wc*qt.tensor(n, Ia) + qt.tensor(Ic, wa/2.*sz) + g*(qt.tensor(at, sm) + qt.tensor(a, sp))
    c_ops = [
        qt.tensor(np.sqrt(kappa*(1+n_th)) * a, Ia),
        qt.tensor(np.sqrt(kappa*n_th) * at, Ia),
        qt.tensor(Ic, np.sqrt(gamma) * sm),
    ]

    psi0 = qt.tensor(qt.fock(dimension, 0), (qt.basis(2, 0) + qt.basis(2, 1)).unit())

    return (H,psi0,tspan,c_ops,[qt.tensor(n, Ia)])

@pytest.fixture(scope='function')
def cavity_setup(dimension):
    kappa = 1.
    eta = 1.5
    wc = 1.8
    wl = 2.
    delta_c = wl - wc
    alpha0 = 0.3 - 0.5j
    tspan = np.linspace(0, 10, 11)

    a = qt.destroy(dimension)
    at = qt.create(dimension)
    n = at * a

    H = delta_c*n + eta*(a + at)
    J = [np.sqrt(kappa) * a]

    psi0 = qt.coherent(dimension, alpha0)
    return (H, psi0, tspan, J, [n])

@pytest.fixture(scope='function')
def qubit_setup(dimension):
    N = int(np.log2(dimension))
   
    # initial state
    state_list = [qt.basis(2, 1)] + [qt.basis(2, 0)] * (N - 1)
    psi0 = qt.tensor(state_list)


    # Energy splitting term
    h = 2 * np.pi * np.ones(N)

    # Interaction coefficients
    Jx = 0.2 * np.pi * np.ones(N)
    Jy = 0.2 * np.pi * np.ones(N)
    Jz = 0.2 * np.pi * np.ones(N)

    # Setup operators for individual qubits
    sx_list, sy_list, sz_list = [], [], []
    for i in range(N):
        op_list = [qt.qeye(2)] * N
        op_list[i] = qt.sigmax()
        sx_list.append(qt.tensor(op_list))
        op_list[i] = qt.sigmay()
        sy_list.append(qt.tensor(op_list))
        op_list[i] = qt.sigmaz()
        sz_list.append(qt.tensor(op_list))

    # Hamiltonian - Energy splitting terms
    H = 0
    for i in range(N):
        H -= 0.5 * h[i] * sz_list[i]


    # Interaction terms
    for n in range(N - 1):
        H += -0.5 * Jx[n] * sx_list[n] * sx_list[n + 1]
        H += -0.5 * Jy[n] * sy_list[n] * sy_list[n + 1]
        H += -0.5 * Jz[n] * sz_list[n] * sz_list[n + 1]

    # dephasing rate
    gamma = 0.02 * np.ones(N)

    # collapse operators
    c_ops = [np.sqrt(gamma[i]) * sz_list[i] for i in range(N)]


    #times
    times = np.linspace(0, 100, 200)
    
    return (H, psi0, times, c_ops, [])

@pytest.mark.parametrize("get_solver",solver_ops,ids=solver_ids)
def test_mesolve(benchmark, model, jc_setup,cavity_setup, qubit_setup, get_solver, request):
    group = request.node.callspec.id
    group = group.split('-')
    benchmark.group = '-'.join(group)

    if(model == 'cavity'):
        setup = cavity_setup
    elif(model == 'jc'):
        setup = jc_setup
    elif(model == 'qubit'):
        setup = qubit_setup

    solver = get_solver(setup)
    result = benchmark(solver,rep)
    return result




