import qutip
import  numpy as np

n = qutip.num(5)

string_form= qutip.QobjEvo([n, 'cos(t)'])

print(string_form)