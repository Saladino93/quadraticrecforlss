"""Compute various spectra and forecasting expressions for specific inputs.

A directory must be specified on the command line. This directory must include:
    - a file 'values.txt' with parameter values for bias and survey
    - a file 'nonlinear_power.txt' with the nonlinear matter power spectrum at the z of interest
    - a file 'linear_power.txt' with the linear matter power spectrum at the z of interest
    - a file 'M.txt' with the M function at z=0

The script dumps a pickle file of a dict with many quantities into 'data_dir/spectra.pickle'.
"""

import sys
import pickle
import itertools

import numpy as np
import scipy.interpolate
import sympy as sp
from sympy.utilities.lambdify  import implemented_function

from quadforlss import estimator as es
import opentext



# Define growth factor for matter domination
# TODO: replace with full growth factor
def D(y):
    return 1./(1.+y)

#################################
# Read input values and spectra
#################################

# Read forecast directory from command line
if len(sys.argv) == 1:
    print('Choose your directory!')
    sys.exit()

direc = str(sys.argv[1])

# Read various input values for forecast
filename = 'values.txt'
values = opentext.get_values(direc+'/'+filename)

# Load linear and nonlinear matter power spectra
nonlinpowerfile = direc+'/nonlinear_power.txt'
linpowerfile = direc+'/linear_power.txt'

K, Pnlin = np.transpose(np.loadtxt(nonlinpowerfile))[0:2,:]
Klin, Plin = np.transpose(np.loadtxt(linpowerfile))[0:2,:]
Plin = np.interp(K, Klin, Plin)

# Load 'M' function and make interpolating function for it
Mtransferfile = 'M.txt'
kM, Mval = np.transpose(np.loadtxt(Mtransferfile))[0:2,:]
Mscipy = scipy.interpolate.interp1d(kM, Mval)

M = implemented_function('M', Mscipy)

# Define min and max k values, making sure M is defined over the full k range
minK = np.maximum(kM.min(), K.min())
maxK = np.minimum(kM.max(), K.max())

# Select k and power spectrum values within range defined above
select = (K>minK)&(K<maxK)
K = K[select]
Plin = Plin[select]
Pnlin = Pnlin[select]

#################################
# Set some parameter values
#################################

# Fetch some values from the input dict
print('Values are, ', values)

minkh = values['minkhanalysis'] #k*h
maxkh = values['maxkhanalysis']

minkhrec = values['minkhrec']
maxkhrec = values['maxkhrec']

bg = values['bgfid']

deltac = values['deltac']
z = values['z']
nbar = values['ngal']

fnl = values['fnlfid']

a1 = values['a1']
a2 = values['a2']

b20 = values['b20']

# Coefficient of f_NL \delta_1 / M term
betaf = 2.*deltac*(bg-1.)

# Total galaxy power spectrum, including Poisson shot noise:
# P_gg^tot = ( b_g + [\beta_f f_NL D(z) / M(k)] )^2 P_nonlin + P_shot
shot = 1/nbar
Ptot = (bg+(betaf*fnl*D(z))/Mscipy(K))**2.*Pnlin+shot
Pnlinsign = Pnlin#(bg+(betaf*fnl*D(z))/Mscipy(K))**2.*Pnlin

# Bias coefficients for quadratic mode-couplings:
#   c_g = b_1 + (7/5) (1/2) b_2 + (7/5) (2/21) (b_1 - 1)
#       TODO: this is different from expression in table in draft
cg = bg+b20/2.*(7./5.)+(7./5.)*(2./21.)*(bg-1)
#   c_s = b_1
cs = bg*1
#   c_t = b_1
#       TODO: this is different from expression in table in draft
ct = bg*1#bg*ct-(2./7.)*(bg-1.)


#################################
# Compute N_{ab} integrals
#################################

# Define k range over which to perform N_{ab} integrals
K_of_interest = np.arange(minkh, maxkh, 0.001)

# Define estimator object
est = es.Estimator(K, Ptot, Plin)

# Define mode-coupling functions
# TODO: Should there be time-dependent factors associated with M?
est.addF('g', 5./7.)
est.addF('s', 0.5*(est.q2/est.q1+est.q1/est.q2)*est.mu)
est.addF('t', (2./7.)*est.mu**2.)
est.addF('b11', 0.5*(1./M(est.q1)+1./M(est.q2)))
est.addF('b01', 0.5*est.mu*est.q1*est.q2*(1./(est.q1**2.*M(est.q2))+1./(est.q2**2.*M(est.q1))))
est.addF('b02', 0.5*(1./(M(est.q1)*M(est.q2))))
est.addF('phiphi', M(sp.sqrt(est.q1**2.+est.q2**2.+2*est.q1*est.q2*est.mu))*(1./M(est.q1))*(1./M(est.q2)))

# Compute and store N_{ab} integrals
est.generateNs(K_of_interest, minkhrec, maxkhrec)



#################################
# Construct dictionary for forecast parameters and outputs
#################################

# Get array of M values
M = Mscipy(est.Krange)

# Get pairs of mode-couplings
values = np.array(['g', 's', 't', 'b11', 'b01', 'b02', 'phiphi'])
listKeys = list(itertools.combinations_with_replacement(list(values), 2))

# Define dict to hold parameters and results
dic = {}

# K range and N_{ab} results
dic['K'] = est.Krange
for a, b in listKeys:
    dic['N'+a+b] = est.getN(a, b)
    dic['N'+b+a] = dic['N'+a+b]

# k ranges for long and short modes
dic['minkh'] = minkh
dic['maxkh'] = maxkh
dic['minkhrec'] = minkhrec
dic['maxkhrec'] = maxkhrec

# Redshift and galaxy number density
dic['z'] = z
dic['ngal'] = nbar

# \beta_f, equal to c_{01} in draft
betaf = 2.*deltac*(bg-1.)
B = betaf
# c_{11}
C = 4*deltac*((deltac/a1**2.)*(b20-2.*(a1+a2)*(bg-1.))-2.*(bg-1.))
# c_{02}
#   TODO: prefactor is different from c_{11} in draft
A = (2./a1**2.)*(deltac*(b20-2*(a1+a2)*(bg-1.))-a1**2.*(bg-1.))+2.*deltac*(bg-1.)

# c_\alpha times appropriate factors of f_NL
dic['kphiphi'] = fnl*bg
dic['kb01'] = fnl*B
dic['kb11'] = fnl*A
dic['kb02'] = fnl**2.*C

# \beta_f = c_{01}
dic['betaf'] = betaf

# d c_{\phi\phi} / d f_NL
dic['dfnlkphiphi'] = bg

# c_g, c_s, c_t
dic['kg'] = cg
dic['ks'] = cs
dic['kt'] = ct

# b_1 and b_2
dic['bg'] = bg
dic['b20'] = b20

# f_NL
dic['fnl'] = fnl
# f_NL factor multiplying \delta_1 on large-scales
dic['bfnllargescales'] = betaf*fnl/M
# derivative of that factor w.r.t f_NL
dic['derbfnllargescales'] = betaf/M

# M, linear power, and shot noise
dic['M'] = M
dic['PL'] = PL
dic['shotnoise'] = shot

# Total gg power spectrum
Cgg = np.interp(est.Krange, K, Ptot)
# Linear matter power
PL = np.interp(est.Krange, K, Plin)
# d C_{gg} / d f_NL
dfnlCgg = 2*(bg+(betaf*fnl*D(z))/M)*(betaf*D(z)/M)*PL

# Construct list of terms in expectation value of \delta_g estimator.
# Each term has the form
#   b_1 c_a f_NL^p(a) N_{gg} / N_{Ga} ,
# where p(a) is the appropriate power of f_NL for term a.
terms = []
for a in values:
    terms += [dic['k'+a]*dic['Ngg']/dic['Ng'+a]]
partial = bg*sum(terms)

# Power spectrum of reconstructed field (using \delta_g estimator)
# is the list of terms above, squared, times P_lin, plus the gg reconstruction noise
Cnn = (partial)**2.*PL + dic['Ngg']
# Derivative of the list of terms w.r.t f_NL
derpartialfnl = bg*dic['Ngg']/dic['Ngphiphi'] \
                + A*dic['Ngg']/dic['Ngb11'] \
                + B*dic['Ngg']/dic['Ngb01'] \
                + 2*fnl*C*dic['Ngg']/dic['Ngb02']
# Derivative of rec. field power spectrum w.r.t f_NL
dfnlCnn = 2*partial*derpartialfnl*PL

# Cross spectrum between g and rec. fields
Cgn = (bg+(betaf*fnl*D(z))/M)*partial*PL
# Derivative of cross spectrum w.r.t f_NL
dfnlCgn = (bg+(betaf*fnl*D(z))/M)*derpartialfnl*PL + (betaf*D(z)/M)*partial*PL

# Add spectra and derivatives to dict
dic['Cgg'] = Cgg
dic['Cnn'] = Cnn
dic['Cgn'] = Cgn
dic['dfnlCgg'] = dfnlCgg
dic['dfnlCgn'] = dfnlCgn
dic['dfnlCnn'] = dfnlCnn

# Input parameter values
dic['values'] = values

#################################
# Save dictionary to disk
#################################

with open(direc+'/data_dir/spectra.pickle', 'wb') as handle:
    pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done')
