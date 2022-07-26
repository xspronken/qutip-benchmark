import qutip
from qutip.solver.mesolve import mesolve

def get_mesolve(setup):
    H,psi0,tspan,c_ops,e_ops = setup
    
    def me_solve(rep):
        for i in range(rep):
            res = mesolve(H, psi0, tspan, c_ops, e_ops)
        return res
    return me_solve