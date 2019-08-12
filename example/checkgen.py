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

kM, Mval = np.transpose(np.loadtxt(Mtransferfile))[0:2,:]
Mscipy = scipy.interpolate.interp1d(kM, Mval)

M = implemented_function('M', Mscipy)

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

shot = 1/nbar
Ptot = (bg+(betaf*fnl)/Mscipy(K))**2.*Pnlin+shot
Pnlinsign = (bg+(betaf*fnl*D(z))/Mscipy(K))**2.*Pnlin

cg = bg+b20/2.*(7./5.)+(7./5.)*(2./21.)*(bg-1)
cs = bg*1
ct = bg*1#bg*ct-(2./7.)*(bg-1.)


############ Begin Calculations

K_of_interest = np.arange(minkh, maxkh, 0.001)

Ptot = scipy.interpolate.interp1d(K, Ptot)
Plin = scipy.interpolate.interp1d(K, Plin)

def _outer_integral(K):
        def _integrand(mu, q):
           modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)
           result = 2*np.pi*q**2./(2*np.pi)**3.
           result *= 2**2*(Plin(q)+Plin(modK_q))**2.
           result /= (2*Ptot(q)*Ptot(modK_q))
           return result*(5/7)**2
        return _integrand

N = []

for K in K_of_interest:
    N += [scipy.integrate.dblquad(_outer_integral(K), minkhrec, maxkhrec, lambda x: -1., lambda x: 1.)[0]]


N = np.array(N)**-1,
print(N)

with open(direc+'/data_dir/spectra.pickle', 'rb') as handle:
    dic = pickle.load(handle, encoding='latin1')

print(dic['Ngg'])

print('Done')
