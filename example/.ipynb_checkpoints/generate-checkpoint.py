from quadforlss import estimator as es

import numpy as np

import pickle

import sympy as sp
from sympy.utilities.lambdify  import implemented_function

import scipy.interpolate


nonlinpowerfile = 'nonlinear_power.txt'
linpowerfile = 'linear_power.txt'

Mtransferfile = 'M.txt'

K, Ptot = np.transpose(np.loadtxt(nonlinpowerfile))[0:2,:]
Klin, Plin = np.transpose(np.loadtxt(linpowerfile))[0:2,:]
Plin = np.interp(K, Klin, Plin)

kM, Mval = np.transpose(np.loadtxt(Mtransferfile))[0:2,:]
Mscipy = scipy.interpolate.interp1d(kM, Mval)

M = implemented_function('M', Mscipy)


minkh, maxkh = 0.005, 0.05
K_of_interest = np.arange(minkh, maxkh, 0.001)

minkhrec, maxkhrec = 0.051, 0.15

est = es.Estimator(K, Ptot, Plin)

est.addF('g', 5./7.)
est.addF('s', 0.5*(est.q2/est.q1+est.q1/est.q2)*est.mu)
est.addF('t', (2./7.)*est.mu**2.)
est.addF('b11', 0.5*(1./M(est.q1)+1./M(est.q2)))
est.addF('b01', 0.5*est.mu*est.q1*est.q2*(1./(est.q1**2.*M(est.q2))+1./(est.q2**2.*M(est.q1))))
est.addF('phiphi', M(sp.sqrt(est.q1**2.+est.q2**2.+2*est.q1*est.q2*est.mu))*(1./M(est.q1))*(1./M(est.q2)))
#est.addF('b20', 1.)
#est.addF('b02', 0.5*(1./(M(est.q1)*M(est.q2))))


est.generateNs(K_of_interest, minkhrec, maxkhrec)

Ngg = est.getN('g', 'g')
Ngb11 = est.getN('g', 'b11')

print('Done')
