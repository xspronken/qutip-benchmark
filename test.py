import qutip 

matrix = qutip.rand_herm(12,density=1)
qobjevo = qutip.QobjEvo([matrix,'cos(t)'])
print(qobjevo)