import sys

from quadforlss import forecasting as fore

from quadforlss import estimator

import opentext

import numpy as np

from matplotlib import pyplot as plt

import scipy

import pickle


def faa(cgg, dercgg):
    A = dercgg/cgg
    tot = 1./2.
    tot *= A**2.
    return tot


if len(sys.argv) == 1:
    print('Choose your directory!')
    sys.exit()

direc = str(sys.argv[1])

filename = 'values.txt'
values = opentext.get_values(direc+'/'+filename)

#Specify your data directory with the N curves
data_dir = direc+'/data_dir/'

#Specify your output plot directory
output = direc+'/pics/'

print('Parameter Values are, ', values)#, 'bgfid ', bgfid, ' bnfid ', bnfid, ' cfid ',cfid, ' fnlfid ', fnlfid)


nonlinpowerfile = direc+'/nonlinear_power.txt'
linpowerfile = direc+'/linear_power.txt'
Mtransferfile = 'M.txt'

K, Pnlin = np.transpose(np.loadtxt(nonlinpowerfile))[0:2,:]
Klin, Plin = np.transpose(np.loadtxt(linpowerfile))[0:2,:]
Plin = np.interp(K, Klin, Plin)

kM, Mval = np.transpose(np.loadtxt(Mtransferfile))[0:2,:]
Mscipy = scipy.interpolate.interp1d(kM, Mval)

sel = np.where(kM > 0.)
kM = kM[sel]
Mval = Mval[sel]

Pnlinscipy = scipy.interpolate.interp1d(K, Pnlin)
Pnlin = Pnlinscipy(kM)
K = kM

b = 1.6
ngal = 1e-2#2*1e-4#1e-2
Nshot = 1/ngal+0.*K
fnlfid = 0.
deltac = 1.42
betaf = 2*deltac*(b-1.)
h = 0.67

#Pnlin *= h**3.
#K *= h**-1.
Mval *= h**-2.

fac = betaf/Mval 
Ptot = (b+fac*fnlfid)**2.*Pnlin+Nshot
dPtot = 2*(b+fac*fnlfid)*fac*Pnlin

Pgg = Ptot
dfnlPgg = dPtot

fnlfid5 = 5.
Ptot5 = (b+fac*fnlfid5)**2.*Pnlin+Nshot

ngal = 2*1e-4
Pgg1 = (b+fac*fnlfid)**2.*Pnlin+1/ngal

plt.title('Baseline 2')
plt.xlabel('$K$ (Mpc$^{-1}$)')
plt.ylabel('Power (Mpc$^{3}$)')
plt.loglog(K, Pgg, label = 'Galaxies Power Spectrum')
#plt.loglog(K, Pgg1, label = 'Galaxies Power Spectrum Baseline 1')
#plt.loglog(K, Ptot5, label = 'Galaxies Power Spectrum, $f_{nl}=5$')
plt.loglog(K, Nshot, label = 'Shot Noise')
plt.legend(loc = 'best', prop = {'size': 6})
plt.rc('grid', linestyle = "-", color = 'black')
plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-')
plt.ylim(bottom = 5)
plt.savefig(output+'galplot.png', dpi = 300)
plt.close()

##################################################


kmin_ksz = 0.0006768356195

sel = np.where((K > kmin_ksz) & (K < 0.11))
K = K[sel]
Pgg = Pgg[sel]
dfnlPgg = dfnlPgg[sel]

f = faa(cgg = Pgg, dercgg = dfnlPgg)

errorgg = f**-0.5

print('Integrated Fisher Error')

Kminforvol = np.min(K)
V = (np.pi)**3./Kminforvol**3.

kmin = np.min(K)
kmax = np.max(K)

print('kmin from ksz, ', kmin_ksz)
print('kmin, ', kmin, ' kmax, ', kmax)

Ks = np.arange(kmin, kmax/1.5, 0.001)

IntegratedFishggVol = np.array([])

physVol = V*1e-9

print('Volume, ', physVol)

V = V*10

for Kmin in Ks:
    IntFish = fore.getIntregratedFisher(K, f, Kmin, kmax, V)
    IntegratedFishggVol = np.append(IntegratedFishggVol, IntFish**-0.5)	

plt.xlabel('$K_{min}$ (Mpc$^{-1}$)')
plt.ylabel('$ Integrated \sigma(f_{nl}) $')
#plt.loglog(Ks, IntegratedFishggVol, label = 'Galaxies Integrated $\sigma(fnl = $'+str(fnlfid)+') for $V = $'+'{:.2E}'.format(physVol)+'$Gpc^3$')
plt.loglog(K, f**-0.5, label = 'PerMode')
plt.legend(loc = 'best', prop = {'size': 6})
plt.rc('grid', linestyle = "-", color = 'black')
plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-')
plt.minorticks_on()
plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2)
plt.ylim(bottom = 5)
plt.savefig(output+'fnlplot.png', dpi = 300)
