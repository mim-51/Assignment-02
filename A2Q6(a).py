from scipy.integrate import tplquad, dblquad
import numpy as np
f1 = lambda z, y, x: x*np.exp(-y)*np.cos(z)
sol=tplquad(f1, 0, 1, lambda x: 0, lambda x: 1 -x*2, lambda x, y: 3, lambda x, y: 4-x2-y*2)[0]
print(sol)
f2 = lambda y, x: (x*y)/np.sqrt(x*2 + y*2 + 1)
sol= dblquad(f2, 0, 1, lambda x: 0, lambda x: 1)[0]
print(sol)