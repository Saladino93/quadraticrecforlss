import numpy as np

import sympy as sp

import mpmath

mpmath.dps = 100

b = np.array([[8.42480818e+00, 8.77083586e-04, 6.55342619e+00, 5.58446078e-01],
 [8.77083586e-04, 4.64692927e-07, 1.47325443e-04, 1.25542447e-05],
 [6.55342619e+00, 1.47325443e-04, 5.86411359e+00, 4.99706740e-01],
 [5.58446078e-01, 1.25542447e-05, 4.99706740e-01, 4.25821946e-02]])


m = mpmath.matrix(b)

print(np.linalg.inv(b)@b)

print(m**-1*m)
