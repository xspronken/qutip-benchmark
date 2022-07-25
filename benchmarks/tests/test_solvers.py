import time
import pytest
import numpy as np
import qutip as qt
from qutip.solver.options import SolverOptions
from . import benchmark_solver



rep = 1

solver_ops = [ getattr(benchmark_solver,_) for _ in dir(benchmark_solver) if _[:3]=="get"]
solver_ids = [ _[4:] for _ in dir(benchmark_solver) if _[:3]=="get"]



@pytest.fixture(params = np.arange(10,161,10,dtype=int).tolist())
def dimension(request): return request.param

@pytest.fixture(params = ["jc", "cavity"])
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


# @pytest.mark.parametrize("get_solver",solver_ops,ids=solver_ids)
# def test_mesolve(benchmark, model, jc_setup,cavity_setup, get_solver, request):
#     group = request.node.callspec.id
#     group = group.split('-')
#     benchmark.group = '-'.join(group)

#     if(model == 'cavity'):
#         setup = cavity_setup
#     elif(model == 'jc'):
#         setup = jc_setup

    
#     solver = get_solver(setup)
#     result = benchmark(solver,rep)
#     return result


@pytest.fixture(params = np.arange(2,11,1,dtype=int).tolist())
def n_qubits(request): return request.param

@pytest.fixture(params = ["qubit"])
def q_model(request): return request.param

@pytest.fixture(scope='function')
def qubit_setup(n_qubits):
    delta = 2 * np.pi
    g = 0.25
   
    eye = []
    for i in range(n_qubits-1):
        eye.append(qt.qeye(2))
    I = qt.tensor(eye)
    
    
    # hamiltonian
    H = delta / 2.0 * qt.tensor(I,qt.sigmax())

    # list of collapse operators
    c_ops = [np.sqrt(g) * qt.tensor(I,qt.sigmaz())]

    # initial state
    qubits = []
    for i in range(n_qubits):
        qubits.append(qt.basis(2,0))
    psi0 = qt.tensor(qubits)


    # times
    tlist = np.linspace(0, 10, 11)
    return (H, psi0, tlist, c_ops, [qt.tensor(I,qt.sigmaz())])

# @pytest.mark.parametrize("get_solver",solver_ops,ids=solver_ids)
# def test_mesolve_qubit(benchmark,q_model, qubit_setup, get_solver, request):
#     group = request.node.callspec.id
#     group = group.split('-')
#     benchmark.group = '-'.join(group)
    
#     if(q_model == 'qubit'):
#         setup = qubit_setup
#     solver = get_solver(setup)
#     result = benchmark(solver,rep)
#     return result