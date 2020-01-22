import sys

from quadforlss import estimator as es

import numpy as np

import pickle

import sympy as sp
from sympy.utilities.lambdify  import implemented_function

import scipy.interpolate
import scipy

import itertools

import opentext

import matplotlib.pyplot as plt

######

def D(y):
        return 1./(1.+y)

######

if len(sys.argv) == 1:
    print('Choose your directory!')
    sys.exit()

direc = str(sys.argv[1])

filename = 'values.txt'
values = opentext.get_values(direc+'/'+filename)

nonlinpowerfile = direc+'/nonlinear_power.txt'
linpowerfile = direc+'/linear_power.txt'

Mtransferfile = direc+'/M.txt'

K, Pnlin = np.transpose(np.loadtxt(nonlinpowerfile))[0:2,:]
Klin, Plin = np.transpose(np.loadtxt(linpowerfile))[0:2,:]
Plin = np.interp(K, Klin, Plin)

kM = K
#, Mval = np.transpose(np.loadtxt(Mtransferfile))[0:2,:]
Mval = 1*K
Mscipy = scipy.interpolate.interp1d(kM, Mval)
#M = implemented_function('M', Mscipy)

minK = np.maximum(kM.min(), K.min())
maxK = np.minimum(kM.max(), K.max())

select = (K>minK)&(K<maxK)
K = K[select]
Plin = Plin[select]
Pnlin = Pnlin[select]

print('Values are, ', values)

minkh = values['minkanalysis'] 
maxkh = values['maxkanalysis']

minkhrec = values['minkrec']
maxkhrec = values['maxkrec']

bg = values['bgfid']

deltac = values['deltac']
z = values['z']
nbar = values['ngal']

fnl = values['fnlfid']

a1 = values['a1']
a2 = values['a2']

b20 = values['b20']

betaf = 2.*deltac*(bg-1.)

bg = 1.
fnl = 0.

shot = 1/nbar
Ptot = (bg+(betaf*fnl)/Mscipy(K))**2.*Pnlin+shot
Pnlinsign = (bg+(betaf*fnl*D(z))/Mscipy(K))**2.*Pnlin


Ktot = K.copy()

cg = bg+b20/2.*(7./5.)+(7./5.)*(2./21.)*(bg-1)
cs = bg*1
ct = bg*1#bg*ct-(2./7.)*(bg-1.)


############ Begin Calculations

K_of_interest = np.arange(minkh, maxkh, 0.001)
Ptot[K>maxkhrec] = np.inf
Ptot = scipy.interpolate.interp1d(K, Ptot)
#Plin[K>maxkhrec] = 0.
Plin = scipy.interpolate.interp1d(K, Plin)
Pnlin = scipy.interpolate.interp1d(K, Pnlin)

print('K of interest, ', K_of_interest)
print('mink for rec', minkhrec, ', maxk for rec', maxkhrec)
print('bg', bg)


import vegas

def _outer_integral(K):
    #@vegas.batchintegrand
    def _integrand(x):
        mu = x[0]
        q = x[1]
        modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)
        result = 2*np.pi*q**2./(2*np.pi)**3.
        result *= (2*5/7)**2*(Plin(q)+Plin(modK_q))**2.
        result /= (2*Ptot(q)*Ptot(modK_q))   
        return result
    return _integrand


minq, maxq = minkhrec, maxkhrec

nitn = 100
neval = 1000

Nvegas = []

for K_i in K_of_interest:
    function = _outer_integral(K_i)
    integ = vegas.Integrator([[-1, 1], [minq, maxq]])#, nhcube_batch = 100)
    result = integ(function, nitn = nitn, neval = neval)
    Nvegas += [result.mean]

Nvegas = np.array(Nvegas)**-1.

print(Nvegas)

'''
bg_n = 1.6#*= 2.
Ptot = (bg_n+(betaf*fnl)/Mscipy(K))**2.*Pnlin(K)+shot
Ptot = np.array(Ptot)
Ptot[K>maxkhrec] = np.inf
Ptot = scipy.interpolate.interp1d(K, Ptot)

Nvegas2 = []

for K_i in K_of_interest:
    function = _outer_integral(K_i)
    integ = vegas.Integrator([[-1, 1], [minq, maxq]])
    result = integ(function, nitn = nitn, neval = neval)
    Nvegas2 += [result.mean]

Nvegas2 = np.array(Nvegas2)**-1.

'''
bg_n = 1.
Ptot = (bg_n+(betaf*fnl)/Mscipy(K))**2.*Pnlin(K)+shot
Ptot = np.array(Ptot)
#Ptot[K>maxkhrec] = 1e8
Ptot = scipy.interpolate.interp1d(K, Ptot)


def _outer_integral(K):
        def step_function(mu, q):
           modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)
           return int(modK_q<maxkhrec)
        def _integrand(mu, q):
           modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)
           result = 2*np.pi*q**2./(2*np.pi)**3.
           result *= (2*5/7)**2*(Plin(q)+Plin(modK_q))**2.
           result /= (2*Ptot(q)*Ptot(modK_q))
           #print(step_function(mu, q))
           return result#*step_function(mu, q)
        return _integrand
N = []
errs = []

for K_i in K_of_interest:
    res = scipy.integrate.dblquad(_outer_integral(K_i), minkhrec, maxkhrec, lambda x: -1., lambda x: 1.) 
    N += [res[0]]
    errs += [res[1]]

#print('Integrals, ', N)
#print('Errs, ', errs)

N = np.array(N)**-1

print(N)

'''

bg_n = 1.6#*= 2.
Ptot = (bg_n+(betaf*fnl)/Mscipy(K))**2.*Pnlin(K)+shot
Ptot = np.array(Ptot)
Ptot = scipy.interpolate.interp1d(K, Ptot)
N2 = []

for K_i in K_of_interest:
    res = scipy.integrate.dblquad(_outer_integral(K_i), minkhrec, maxkhrec, lambda x: -1., lambda x: 1.)
    N2 += [res[0]]

N2 = np.array(N2)**-1

print('Prediction is, ', (bg_n/bg)**4)
print(N2/N)
'''

xnew = K_of_interest #np.linspace(K_of_interest.min(), K_of_interest.max(), 15)

#from scipy.interpolate import make_interp_spline, BSpline
#spl = make_interp_spline(K_of_interest, Nvegas, k = 3)
#Nvegas = spl(xnew)

#spl2 = make_interp_spline(K_of_interest, Plin(K_of_interest), k = 3)
Plin = Plin(xnew)#spl2(xnew)

plt.yscale('log')
plt.plot(K_of_interest, N, label = 'N')
plt.plot(xnew, Nvegas, label = 'N vegas')
#plt.plot(K_of_interest, Nvegas2, label = 'N vegas2')
#plt.plot(K_of_interest, N2, label = 'N2')
plt.plot(xnew, Plin)
plt.legend()
plt.savefig('pss.png', dpi = 300)
plt.close()


'''

with open(direc+'/data_dir/spectra.pickle', 'rb') as handle:
    dic = pickle.load(handle, encoding='latin1')

print(dic['Ngg'])

Ptot = (bg+(betaf*fnl)/Mscipy(K))**2.*Pnlin(K)+shot
np.savetxt(direc+'/totpower.txt', np.c_[Ktot, Ptot])
'''

print('Done')
