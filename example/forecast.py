"""Compute Fisher forecast for f_NL.
"""

import sys

from quadforlss import forecasting as fore
from quadforlss import estimator
import opentext

import numpy as np
from matplotlib import pyplot as plt
import scipy
import pickle

# Diagonal Fisher matrix element, from paper
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
#, 'bgfid ', bgfid, ' bnfid ', bnfid, ' cfid ',cfid, ' fnlfid ', fnlfid)

# Read output of generate.py from pickle file
with open(direc+'/data_dir/spectra.pickle', 'rb') as handle:
    dic = pickle.load(handle, encoding='latin1')

# Define spectra, derivatives w.r.t f_NL, and other stuff
P_L = dic['PL']
Pgg = dic['Cgg']
Pnn = dic['Cnn']
Pgn = dic['Cgn']
dfnlPgg = dic['dfnlCgg']
dfnlPgn = dic['dfnlCgn']
dfnlPnn = dic['dfnlCnn']
N = dic['Ngg']
K = dic['K']
keys = dic['values']
M = dic['M']
bgfid = dic['bg']
fnl = dic['fnl']

nbar = values['ngal']
fnlfid = values['fnlfid']

# Galaxy shot power, as 1/ngal
shot = 1./dic['ngal']+0.*Pgg


#################################
# Plot input signal and noise spectra
#################################

# Save plot of P_{gg}, P_{gn}, and P_{nn}
fig, ax = plt.subplots( nrows=1, ncols=1 )
#plt.xlim(0.01, 0.1)
#plt.ylim(1e1, 1e8)
plt.xlabel('$K$ $(Mpc^{-1})$')
plt.ylabel('$P$ $(Mpc^{3})$')
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
plt.xlabel('$K$ $(Mpc^{-1})$')
plt.ylabel('$P$ $(Mpc^{3})$')
ax.plot(K, K*0+shot, label = 'Shot Noise')
ax.plot(K, N, label = 'Nnn, n = growth est')
ax.plot(K, P_L, label = 'P Linear')
ax.legend(loc = 'best', prop = {'size': 6})
fig.savefig(output+'signal_noise_powers_forecast_fid_fnl'+str(fnlfid)+'.png', dpi = 300)
plt.close(fig)

# Set kmin and kmax for reconstruction
minkrec = values['minkrec']
maxkrec = values['maxkrec']

# Define \alpha such that M(k) = \alpha * k^2 * T(k)
alpha = 42328969.6526119
# Define \Delta v = \frac{4\pi}{3} ( k_max^3 - k_min^3)
deltak = maxkrec-minkrec
deltaV = 4*np.pi/3*(maxkrec**3.-minkrec**3.)

# Define variables to hold sums of various bias terms
somma = 0.
somma_fnl = 0.
somma_other = 0.

# Dict defining colors for plots
color = {'g': 'red', 's': 'blue', 't': 'green', 'phiphi': 'black', 'b01': 'orange',
        'b02': 'yellow', 'b11': 'cyan'}

# Plot mean-field biases
fig, ax = plt.subplots( nrows = 1, ncols = 1 )
plt.rc('font', size = 8)
plt.title('Biases induced in the mean field reconstruction case of $f_{nl}=$'+str(fnl)
            +', $nbar = $'+str(nbar)+ ' $b_g=$'+str(bgfid) )
plt.xlabel('$K$ $(Mpc^{-1})$')
plt.ylabel('$Bias$')
for a in keys.astype(str):
    factor = 1.
    sign = ''
    if a in np.array(['s', 'b01', 'b02']):
        factor = -1.
        sign = '-'
    somma += bgfid*N/dic['N'+a+'g']*dic['k'+a]
    if a in np.array(['phiphi', 'b01', 'b02', 'b11']):
        somma_fnl += bgfid*N/dic['N'+a+'g']*dic['k'+a]
    elif a in np.array(['g', 's', 't']):
        somma_other += bgfid*N/dic['N'+a+'g']*dic['k'+a]
    ax.loglog(K, factor*bgfid*N/dic['N'+a+'g']*dic['k'+a], color = color[a], label = sign +a +' term')

ax.loglog(K, somma, label = 'Biases Sum')
ax.loglog(K, somma_fnl, label = 'fnl Biases Sum')
intergration = 0.67*0.785278

# Plot analytical approximations for different terms
ax.loglog(K, K*0.+bgfid*dic['kb01']*4*np.pi*7/10*(1/alpha)*(deltak/deltaV+intergration), ls = '--', color = color['b01'], label = 'Analytic Approx b01')
ax.loglog(K, bgfid*dic['kphiphi']*7./5.*1/M, ls = '--', color = color['phiphi'], label = 'Analytic Approx phiphi')
ax.loglog(K, bgfid*dic['kb11']*7./10.*(1/M+4*np.pi/deltaV*1/alpha*deltak), ls = '--', color = color['b11'], label = 'Analytic Approx b11')
ax.loglog(K, -bgfid*dic['kb02']*14./5.*2*np.pi*1./alpha*(1/deltaV)*1/M*intergration, color = color['b02'], ls = '--', label = 'Analytic Approx b02' )
ax.loglog(K, bgfid*dic['derbfnllargescales']*fnl, color = 'pink', label = '$f_{nl}$ signal in the galaxy field on large scales')
ax.legend(loc = 'best', prop = {'size': 6})
fig.savefig(output+'biases_forecast_fid_fnl'+str(fnlfid)+'.png', dpi = 300)
plt.close(fig)

# Plot some more bias stuff
fig, ax = plt.subplots( nrows = 1, ncols = 1 )
plt.rc('font', size = 8)
plt.title('Biases induced in the mean field reconstruction case of $f_{nl}=$'+str(fnl)+', $nbar = $'+str(nbar)+ ' $b_g=$'+str(bgfid) )
plt.xlabel('$K$ $(Mpc^{-1})$')
plt.ylabel('$Bias$')
ax.loglog(K, abs(dic['derbfnllargescales']/bgfid**2.-somma_fnl/somma_other))
ax.legend(loc = 'best', prop = {'size': 6})
fig.savefig(output+'contamination_difference_forecast_fid_fnl'+str(fnlfid)+'.png', dpi = 300)
plt.close(fig)


#################################
# Do Fisher computations and plot output
#################################

# Set parameter indices for forecast
keyfnl = 'fnl'
el1, el2 = keyfnl, keyfnl

# Set redshift for forecasts
# This isn't actually used, so it is commented out
# z = 0.

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
print(r)

# Alternative versions of forecasts
# TODO: sort out this code: which version of forecast to use?
f = faa(r = r, cgg = Pgg, cgn = Pgn, cnn = Pnn,
        dercgg = dfnlPgg, dercgn = dfnlPgn, dercnn = dfnlPnn)

forecast = f.copy()

f = faa(r = 0*Pgg, cgg = Pgg, cgn = Pgn, cnn = Pnn,
        dercgg = dfnlPgg, dercgn = 0., dercnn = 0.)

# Compute unmarginalized errorbars on f_NL from Fisher elements
errorgg = forecastgg**-0.5
error = forecast**-0.5

# Plot unmarginalized f_NL errorbars, and forecasted gn cross-correlation coeff.
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 10))

ax3.set_xlabel('$K$ ($h$Mpc$^{-1}$)')
ax3.set_ylabel('$ \sigma(f_{nl}) $')
ax3.loglog(K, error, label = 'Sigma(fnl = '+str(fnlfid)+')')
ax3.loglog(K, errorgg, label = 'Galaxy Only Sigma(fnl = '+str(fnlfid)+')')
ax3.legend(loc = 'best', prop = {'size': 6})

ax1.set_xlabel('$K$ (Mpc$^{-1}$)')
ax1.set_ylabel('$ r $')
ax1.set_xscale('log')
ax1.plot(K, r, label = 'Corr, fnl='+str(fnlfid))
ax1.legend(loc = 'best', prop = {'size': 6})


# Now, compute Fisher error integrated over all k
print('Integrated Fisher Error')

# Compute comoving survey volume based on k_min
# TODO: we should set V externally instead
Kminforvol = np.min(K)
V = (np.pi)**3./Kminforvol**3.

FisherPerMode = forecast
FisherPerModegg = forecastgg

# Set range of k_min values to explore in forecast
kmin = np.min(K)
kmax = np.max(K)
print('kmin, ', kmin, ' kmax, ', kmax)
Ks = np.arange(kmin, kmax/1.5, 0.001)

# For each k_min, compute integrated Fisher matrix, both with and without
# reconstructed field included, keeping V fixed, and make arrays of
# unmarginalized f_NL errorbar
IntegratedFish = np.array([])
IntegratedFishgg = np.array([])
IntegratedFishVol = np.array([])
IntegratedFishggVol = np.array([])
for Kmin in Ks:
    IntFish = fore.getIntregratedFisher(K, FisherPerMode, Kmin, kmax, V)
    IntegratedFish = np.append(IntegratedFish, IntFish**-0.5)
    IntFish = fore.getIntregratedFisher(K, FisherPerModegg, Kmin, kmax, V)
    IntegratedFishgg = np.append(IntegratedFishgg, IntFish**-0.5)

# Compute different volume for each k_min
# TODO: again, we should keep V fixed from an external input
h = 0.67
V = (2*np.pi)**3/kmin**3.#h**3*100*10**9
V = (np.pi)**3/kmin**3
print('Volume, ', (V/1e9))

# For each k_min, compute integrated Fisher matrix, both with and without
# reconstructed field included, also varying V with k_min
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


# Plot forecasts from integrated Fisher elements
ax2.set_xlabel('$K_{min}$ (Mpc$^{-1}$)')
ax2.set_ylabel('$ Integrated \sigma(f_{nl}) $')
ax2.loglog(Ks, IntegratedFishVol, label = 'Integrated $\sigma(fnl = $'+str(fnlfid)+') for $V = $'+'{:.2E}'.format(V)+'$h^{-3}Mpc^3$')
ax2.loglog(Ks, IntegratedFishggVol, label = 'Galaxies Integrated $\sigma(fnl = $'+str(fnlfid)+') for $V = $'+'{:.2E}'.format(V)+'$h^{-3}Mpc^3$')
ax2.legend(loc = 'best', prop = {'size': 6})

# Plot ratio of g+n and g-only integrated Fisher elements
# as function of k_min
ax4.set_xlabel('$K_{min}$ (Mpc$^{-1}$)')
ax4.set_ylabel('$Fraction$')
ax4.plot(Ks, Ks*0.+1.)
ax4.plot(Ks, IntegratedFish/IntegratedFishgg, label = 'Integrated $\sigma_{combined/gonly}(fnl = $'+str(fnlfid)+')')
ax4.legend(loc = 'best', prop = {'size': 6})
plt.subplots_adjust(hspace = 0.2, wspace = 0.5)
fig.savefig(output+'plots.png', dpi = 300)
plt.close(fig)
