import qutip 
import numpy as np
from scipy.sparse import rand
size =5
n = qutip.num(size)
t = 1.0
a = qutip.destroy(size)
ad = qutip.create(size)
n = qutip.num(size)
Id = qutip.qeye(size)

matrix = qutip.rand_ket(5, density=1)


H = np.random.random((size, size)) + 1j*np.random.random((size, size))
S = qutip.Qobj(H + H.T.conj())

ofdiag = np.random.rand(size-1) + 1j*np.random.rand(size-1)
diag = np.random.rand(size) + 1j*np.random.rand(size)

I = qutip.Qobj(np.diag(ofdiag, k=-1)
        + np.diag(diag, k=0)
        + np.diag(ofdiag.conj(), k=1))



string_form = qutip.QobjEvo([n, [a + ad, "cos(t)"]])

print(string_form(2))