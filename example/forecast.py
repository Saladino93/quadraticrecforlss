"""Compute Fisher forecast for f_NL.
"""

import sys
import pickle

import numpy as np
import scipy
from matplotlib import pyplot as plt

from quadforlss import forecasting as fore
from quadforlss import estimator
import opentext

# Diagonal Fisher matrix element, from Eq. (17) in draft
def faa(r, cgg, cgn, cnn, dercgg, dercgn, dercnn):
    A = dercgg/cgg-2*r**2.*dercgn/cgn
    A = A**2.
    B = 2*r**2.*(1.-r**2.)*(dercgn/cgn)**2.
    C = 2*r**2.*dercnn/cnn
    C *= (dercgg/cgg-2*dercgn/cgn)
    D = dercnn/cnn
    D = D**2.
    tot = 1./(2*(1.-r**2.)**2.)
    tot *= (A+B+C+D)
    return tot


#################################
# Read input values, spectra, and N_{ab} matrix from disk
#################################

# Read forecast directory from command line
if len(sys.argv) == 1:
    print('Choose your directory!')
    sys.exit()

direc = str(sys.argv[1])

# Read various input values for forecast
filename = 'values.txt'
values = opentext.get_values(direc+'/'+filename)

# Set directories holding data and plots
data_dir = direc+'/data_dir/'
output = direc+'/pics/'

# Print input parameter values
print('Parameter Values are, ', values)

# Read output of generate.py from pickle file
with open(direc+'/data_dir/spectra.pickle', 'rb') as handle:
    dic = pickle.load(handle, encoding='latin1')

# Define spectra
P_L = dic['PL']
Pgg = dic['Cgg']
Pnn = dic['Cnn']
Pgn = dic['Cgn']
dfnlPgg = dic['dfnlCgg']
dfnlPgn = dic['dfnlCgn']
dfnlPnn = dic['dfnlCnn']
N = dic['Ngg']

# Define k range, galaxy number density, shot power, and f_NL value
K = dic['K']
nbar = values['ngal']
shot = 1./dic['ngal']+0.*Pgg
fnlfid = values['fnlfid']



#################################
# Plot input signal and noise spectra
#################################

# Save plot of P_{gg}, P_{gn}, and P_{nn}
fig, ax = plt.subplots( nrows=1, ncols=1 )
#plt.xlim(0.01, 0.1)
#plt.ylim(1e1, 1e8)
plt.xlabel('$K$ $(h Mpc^{-1})$')
plt.ylabel('$P$ $(h^{-3} Mpc^{3})$')
ax.plot(K, Pgg, label = 'Pgg for fnl='+str(fnlfid))
ax.plot(K, Pnn, label = 'Pnn, n = growth est')
ax.plot(K, Pgn, label = 'Pgn')
ax.legend(loc = 'best', prop = {'size': 6})
fig.savefig(output+'powers_forecast_fid_fnl'+str(fnlfid)+'.png', dpi = 300)
plt.close(fig)

# Save plot of P_shot, N_{nn}, and P_linear
fig, ax = plt.subplots( nrows=1, ncols=1 )
#plt.xlim(0.01, 0.1)
#plt.ylim(1e1, 1e8)
plt.xlabel('$K$ $(h Mpc^{-1})$')
plt.ylabel('$P$ $(h^{-3} Mpc^{3})$')
ax.plot(K, K*0+shot, label = 'Shot Noise')
ax.plot(K, N, label = 'Nnn, n = growth est')
ax.plot(K, P_L, label = 'P Linear')
ax.legend(loc = 'best', prop = {'size': 6})
fig.savefig(output+'signal_noise_powers_forecast_fid_fnl'+str(fnlfid)+'.png', dpi = 300)
plt.close(fig)


#################################
# Do Fisher computations
#################################

# Set parameter indices for forecast
keyfnl = 'fnl'
el1, el2 = keyfnl, keyfnl

# Some alternate forecasts
'''
mu = 0 # our results do not depend on mu
#forecast0 = fore.getFisherpermodefnlfid0(el1, el2, k = K, mu = mu,
                Pgg = Pgg, Pnn = Pnn, Pgn = Pgn,
                cfid = cfid, bgfid = bgfid, bnfid = bnfid, kappafid = kappafid)
forecast = fore.getFisherpermode(el1, el2, k = K, mu = mu,
                Pgg = Pgg, Pnn = Pnn, Pgn = Pgn, PLfid = P_L,
                fnlfid = fnlfid, cfid = cfid, bgfid = bgfid, bnfid = bnfid, kappafid = kappafid)
forecastgg = fore.getFisherpermodeggonly(el1, el2, k = K, mu = mu,
                Pgg = Pgg, PLfid = P_L,
                fnlfid = fnlfid, cfid = cfid, bgfid = bgfid)
'''

# Define more parameters needed for forecasts
PL = P_L
bg = dic['bg']
M = dic['M']
betaf = dic['betaf']
fnl = dic['fnl']
kappafid = 1.

# Linear growth factor during matter domination
# TODO: change to full growth factor
def D(y):
    return 1/(1+y)

# Redshift for forecasts
z = 0.

#bn = 0.19

# Compute Fisher element for f_NL, including reconstructed field
forecast = fore.getcompleteFisher(cgg = Pgg, cgn = Pgn, cnn = Pnn,
                                    acgg = dfnlPgg, acgn = dfnlPgn, acnn = dfnlPnn)
# Compute Fisher element for f_NL, without reconstructed field.
# In practice, set derivatives involving rec. field to zero, and set spectra
# to numerically small values (why?)
forecastgg = fore.getcompleteFisher(cgg = Pgg, cgn = 1.e-4, cnn = 1.e-4,
                                    acgg = dfnlPgg, acgn = 0., acnn = 0.)
                                    #put only derivatives to zero

# Cross-correlation coefficient between g and n fields
r = Pgn/np.sqrt(Pgg*Pnn)
#dfnlPgn = 0.
#dfnlPgg = 0.
#r = 0.*K


# Alternative versions of forecasts
# TODO: sort out this code: which version of forecast to use?
f = faa(r = r, cgg = Pgg, cgn = Pgn, cnn = Pnn,
        dercgg = dfnlPgg, dercgn = dfnlPgn, dercnn = dfnlPnn)

forecast = f.copy()

f = faa(r = 0, cgg = Pgg, cgn = Pgn, cnn = Pnn,
        dercgg = dfnlPgg, dercgn = 0., dercnn = 0.)
#forecastgg = f.copy()

# Compute unmarginalized errorbars on f_NL from Fisher elements
errorgg = forecastgg**-0.5
error = forecast**-0.5

# Now, compute Fisher error integrated over all k
print('Integrated Fisher Error')

# Compute comoving survey volume based on k_min
Kminforvol = np.min(K)
V = (2.*np.pi)**3./Kminforvol**3.

FisherPerMode = forecast
FisherPerModegg = forecastgg

# Set range of k_min values to explore in forecast
kmin = np.min(K)
kmax = np.max(K)
print('kmin, ', kmin, ' kmax, ', kmax)
Ks = np.arange(kmin, kmax/1.5, 0.001)

# For each k_min, compute integrated Fisher matrix, both with and without
# reconstructed field included, keeping V fixed
IntegratedFish = np.array([])
IntegratedFishgg = np.array([])
for Kmin in Ks:
    IntFish = fore.getIntregratedFisher(K, FisherPerMode, Kmin, kmax, V)
    IntegratedFish = np.append(IntegratedFish, IntFish**-0.5)
    IntFish = fore.getIntregratedFisher(K, FisherPerModegg, Kmin, kmax, V)
    IntegratedFishgg = np.append(IntegratedFishgg, IntFish**-0.5)

# Compute different volume for each k_min
# TODO: which V is correct here?
h = 0.67
V = (2*np.pi)**3/kmin**3.#h**3*100*10**9
V = (np.pi)**3/kmin**3/2.
print('Volume, ', ((2*np.pi)**3/(kmin*h*1e3)**3.))

# For each k_min, compute integrated Fisher matrix, both with and without
# reconstructed field included, also varying V with k_min
IntegratedFishVol = np.array([])
IntegratedFishggVol = np.array([])
for Kmin in Ks:
    IntFish = fore.getIntregratedFisher(K, FisherPerMode, Kmin, kmax, V)
    IntegratedFishVol = np.append(IntegratedFishVol, IntFish**-0.5)
    IntFish = fore.getIntregratedFisher(K, FisherPerModegg, Kmin, kmax, V)
    IntegratedFishggVol = np.append(IntegratedFishggVol, IntFish**-0.5)

# Save different Fisher versions and cross-corr coefficient to text files
text = ''#'zerocorr'
np.savetxt(data_dir+'singlefisher'+text+'.txt', np.c_[K, error, errorgg])
np.savetxt(data_dir+'integratedfisher'+text+'.txt', np.c_[Ks, IntegratedFishVol, IntegratedFishggVol])
np.savetxt(data_dir+'r.txt', np.c_[K, r])


#################################
# Plot outputs
#################################

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 10))

# Plot \sigma(f_NL) per mode, as function of k
ax3.set_xlabel('$K$ ($h$Mpc$^{-1}$)')
ax3.set_ylabel('$ \sigma(f_{nl}) $')
ax3.loglog(K, error, label = 'Sigma(fnl = '+str(fnlfid)+')')
ax3.loglog(K, errorgg, label = 'Galaxy Only Sigma(fnl = '+str(fnlfid)+')')
ax3.legend(loc = 'best', prop = {'size': 6})
#fig.savefig(output+'forecast_fid_fnl'+str(fnlfid)+'.png', dpi = 300)
#plt._ax.setxlim(1e-3, 1e-2)
#plt.close(fig)

# Plot g-n cross-correlation coefficient
#fig, ax = plt.subplots(nrows=1, ncols=1 )
#plt.xlim(0.01, 0.1)
#plt.ylim(1e1, 1e8)
ax1.set_xlabel('$K$ ($h$Mpc$^{-1}$)')
ax1.set_ylabel('$ r $')
ax1.plot(K, r, label = 'Corr, fnl='+str(fnlfid))
ax1.legend(loc = 'best', prop = {'size': 6})


# Plot integrated Fisher elements (with varying V) as function of k_min
ax2.set_xlabel('$K_{min}$ ($h$Mpc$^{-1}$)')
ax2.set_ylabel('$ Integrated \sigma(f_{nl}) $')
ax2.loglog(Ks, IntegratedFishVol,
            label = 'Integrated $\sigma(fnl = $'+str(fnlfid)
                        +') for $V = $'+'{:.2E}'.format(V)+'$h^{-3}Mpc^3$')
ax2.loglog(Ks, IntegratedFishggVol,
            label = 'Galaxies Integrated $\sigma(fnl = $'+str(fnlfid)
                        +') for $V = $'+'{:.2E}'.format(V)+'$h^{-3}Mpc^3$')
ax2.legend(loc = 'best', prop = {'size': 6})

# Plot ratio of g+n and g-only integrated Fisher elements (with fixed V)
# as function of k_min
ax4.set_xlabel('$K_{min}$ ($h$Mpc$^{-1}$)')
ax4.set_ylabel('$Fraction$')
ax4.plot(Ks, Ks*0.+1.)
ax4.plot(Ks, IntegratedFish/IntegratedFishgg,
        label = 'Integrated $\sigma_{combined/gonly}(fnl = $'+str(fnlfid)+')')
ax4.legend(loc = 'best', prop = {'size': 6})
plt.subplots_adjust(hspace = 0.2, wspace = 0.5)
fig.savefig(output+'plots.png', dpi = 300)
plt.close(fig)
