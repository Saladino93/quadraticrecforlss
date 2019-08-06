import sys

from quadforlss import forecasting as fore

from quadforlss import estimator

import opentext

import numpy as np

from matplotlib import pyplot as plt

import scipy

import sympy as sp

def fab(cgg, dera_cgg, derb_cgg):
    A = dera_cgg/cgg
    B = derb_cgg/cgg
    tot = 1./2.
    tot *= A*B
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

bg = 1.6
ngal = 1e-2#2*1e-4#1e-2
Nshot = 1/ngal+0.*K
fnlfid = 0.
deltac = 1.42
betaf = 2*deltac*(bg-1.)
h = 0.67

Pnlin *= h**3.
K *= h**-1.
Mval *= h**-2.

fac = betaf/Mval 
Ptot = (bg+fac*fnlfid)**2.*Pnlin+Nshot
dPtot = 2*(bg+fac*fnlfid)*fac*Pnlin

Pgg = Ptot
dfnlPgg = dPtot

fnlfid5 = 5.
Ptot5 = (bg+fac*fnlfid5)**2.*Pnlin+Nshot

ngal = 2*1e-4
Pgg1 = (bg+fac*fnlfid)**2.*Pnlin+1/ngal


#### Define symbols

b, fnl, nbar, Pnl, fnlfunc = sp.symbols('b fnl nbar Pnl fnlfunc')

func = 2*(b-1.)*deltac*fnlfunc
bias = b+fnl*func
P_total = bias**2.*Pnl+1/nbar

derb_P_s = sp.diff(P_total, b)
derb_P = sp.lambdify([b, fnl, nbar, Pnl, fnlfunc], derb_P_s, 'numpy')

derfnl_P_s = sp.diff(P_total, fnl)
derfnl_P = sp.lambdify([b, fnl, nbar, Pnl, fnlfunc], derfnl_P_s, 'numpy')


derbP = derb_P(bg, fnlfid, ngal, Pnlin, 1./Mval)
derfnlP = derfnl_P(bg, fnlfid, ngal, Pnlin, 1./Mval)


derbP2 = 2*Pnlin*(bg+fnlfid*2.*deltac*(bg-1.)/Mval)*(1+2*deltac/Mval)
derfnlP2 = 2*Pnlin*(bg+fnlfid*2.*deltac*(bg-1.)/Mval)*2.*deltac*(bg-1.)/Mval

fbb = fab(Pgg, derbP, derbP)
ffnlfnl = fab(Pgg, derfnlP, derfnlP)
ffnlb = fab(Pgg, derfnlP, derbP)

kmin_ksz = 0.0006768356195
kmax_ksz = 0.11

sel = np.where((K > kmin_ksz) & (K < kmax_ksz))
K = K[sel]
kmin = np.min(K)
kmax = np.max(K)
V = (np.pi)**3./kmin**3.
Ks = np.arange(kmin, kmax/1.5, 0.001)
print('kmin, ', kmin, ' V,(Gpc^3) ', V/1e9)
fbb = fbb[sel]
ffnlfnl = ffnlfnl[sel]
ffnlb = ffnlb[sel]

func_fbb = scipy.interpolate.interp1d(K, fbb)
func_ffnlfnl = scipy.interpolate.interp1d(K, ffnlfnl)
func_ffnlb = scipy.interpolate.interp1d(K, ffnlb)

mat = np.zeros((2, 2))

fint = []

errfnlmarg = []

for k in Ks:
    mat[0, 0] = scipy.integrate.quad(lambda x: func_fbb(x)*x**2., k, kmax)[0]   
    mat[1, 1] = scipy.integrate.quad(lambda x: func_ffnlfnl(x)*x**2., k, kmax)[0]    
    mat[0, 1] = scipy.integrate.quad(lambda x: func_ffnlb(x)*x**2., k, kmax)[0]
    mat[1, 0] = mat[0, 1]
    mat *= V/(2*np.pi**2.)
    #fint += [mat]
    invmat = np.linalg.inv(mat)
    el = np.sqrt(invmat[1, 1])
    #print(invmat)
    errfnlmarg += [el]

''' 
fint = np.array(fint)

invf = []

for a in fint:
    invmat = np.linalg.inv(a)
    invf += [invmat]

invf = np.array(invf)

errfnlmarg = []

for a in invf:
    el = np.sqrt(a[1, 1])
    errfnlmarg += [el]
'''

errfnlmarg = np.array(errfnlmarg)


f = fab(cgg = Pgg, dera_cgg = dfnlPgg, derb_cgg = dfnlPgg)
f = f[sel]
func_f = scipy.interpolate.interp1d(K, f)

fint = []

for k in Ks:
    mat = scipy.integrate.quad(lambda x: func_f(x)*x**2., k, kmax)[0]
    mat *= V/(2*np.pi**2.)
    fint += [mat]

fint = np.array(fint)
invf = fint**-1.

'''

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

f = fab(cgg = Pgg, dera_cgg = dfnlPgg, derb_cgg = dfnlPgg)

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
'''
plt.xlabel('$K_{min}$ (Mpc$^{-1}$)')
plt.ylabel('$ Integrated \sigma(f_{nl}) $')
plt.loglog(Ks, invf, label = 'Galaxies Integrated $\sigma(fnl = $'+str(fnlfid)+') for $V = $'+'{:.2E}'.format(V)+'$Gpc^3$')
plt.loglog(Ks, errfnlmarg, label = 'Marginalized')
#plt.loglog(K, f**-0.5, label = 'PerMode')
plt.legend(loc = 'best', prop = {'size': 6})
plt.rc('grid', linestyle = "-", color = 'black')
plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-')
plt.minorticks_on()
plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2)
plt.ylim(bottom = 5)
plt.savefig(output+'fnlplot.png', dpi = 300)
