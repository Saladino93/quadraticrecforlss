"""Compute various spectra and forecasting expressions for specific inputs.

A directory must be specified on the command line. This directory must include:
    - a file 'values.txt' with parameter values for bias and survey
    - a file 'nonlinear_power.txt' with the nonlinear matter power spectrum at the z of interest
    - a file 'linear_power.txt' with the linear matter power spectrum at the z of interest
    - a file 'M.txt' with the M function at the z of interest

The script dumps a pickle file of a dict with many quantities into 'data_dir/spectra.pickle'.
"""

import sys

from quadforlss import estimator as es

import numpy as np

import pickle

import sympy as sp
from sympy.utilities.lambdify  import implemented_function

import scipy.interpolate

import itertools

import yaml


#################################
# Read input values and spectra
#################################

# Read forecast directory from command line
if len(sys.argv) == 1:
    print('Choose your configuration file!')
    sys.exit()

# Read configuration file with yaml
values_file = str(sys.argv[1])
with open(values_file, 'r') as stream:
    data = yaml.safe_load(stream)
values = data

# Get directories
direc = values['name']
base_dir = values['file_config']['base_dir']
data_dir = values['file_config']['data_dir']

dic_name = values['file_config']['dic_name']

direc = base_dir+direc+'/'

#################

# Get file names for nonlinear power spectrum, linear spectrum, and M(k)
nonlinpowerfile = direc+data_dir+values['file_config']['nonlinear_power_name']
linpowerfile = direc+data_dir+values['file_config']['linear_power_name']
Mtransferfile = direc+data_dir+values['file_config']['M_name']

# Read in P_nonlin and P_lin
K, Pnlin = np.transpose(np.loadtxt(nonlinpowerfile))[0:2,:]
Klin, Plin = np.transpose(np.loadtxt(linpowerfile))[0:2,:]
Plin = np.interp(K, Klin, Plin)

# Read in M
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
# print('Values are, ', values)

minkh = values['analysis_config']['mink_analysis']
maxkh = values['analysis_config']['maxk_analysis']

minkhrec = values['analysis_config']['mink_reconstruction']
maxkhrec = values['analysis_config']['maxk_reconstruction']

z = values['data_creation_config']['z']

''' GRAVITY MODEL PARAMETERS '''

deltac = values['survey_config']['gravity_dynamics_pars']['deltac']
a1 = values['survey_config']['gravity_dynamics_pars']['a1']
a2 = values['survey_config']['gravity_dynamics_pars']['a2']

''' BIAS PARAMETERS '''

b10 = values['survey_config']['tracer_properties']['biases']['b10']
b01 = values['survey_config']['tracer_properties']['biases']['b01']
b11 = values['survey_config']['tracer_properties']['biases']['b11']
b02 = values['survey_config']['tracer_properties']['biases']['b02']
b20 = values['survey_config']['tracer_properties']['biases']['b20']
bs2 = values['survey_config']['tracer_properties']['biases']['bs2']

b10 = float(b10)
b20 = float(b20)

betaf = 2.*deltac*(b10-1.)


##### LATER FOR FORECAST

variables_list = values['forecast_config']['variables_list']



# If nonlinear bias values aren't specified, use theory predictions
if bs2 == '':
   print('Using theory value for bs2!')
   bs2 = -2./7.*(b10-1)
else:
   bs2 = float(bs2)

if b01 == '':
   print('Using theory value for b01!')
   b01 = betaf
else:
   b01 = float(b01)

if b11 == '':
   print('Using theory value for b11!')
   b11 = (2./a1)*(deltac*(b20-2*(a1+a2)*(b10-1.))-a1**2.*(b10-1.))+2.*deltac*(b10-1.)
else:
   b11 = float(b11)

if b02 == '':
   print('Using theory value for b02!')
   b02 = 4*deltac*((deltac/a1**2.)*(b20-2.*(a1+a2)*(b10-1.))-2.*(b10-1.))
else:
   b02 = float(b02)

''' OTHER TRACER PROPERTIES '''

mhalo = values['survey_config']['tracer_properties']['mhalo']
nhalo = values['survey_config']['tracer_properties']['nhalo']
fnl = values['survey_config']['primordial_pars']['fnl']

mhalo = float(mhalo)
nhalo = float(nhalo)
fnl = float(fnl)

''' MODEL INPUT TOTAL POWER SPECTRUM

The total power spectrum will be given by the square b10+betaf*fnl/M
times the NON linear power spectrum plus the shot noise contribution of the tracer

'''

shot = 1/nhalo

Ptot = (b10+(betaf*fnl)/Mscipy(K))**2.*Pnlin+shot

Pnlinsign = (b10+(betaf*fnl)/Mscipy(K))**2.*Pnlin

''' CREATE ESTIMATOR OBJECT FOR NOISE CALCULATIONS '''

vegas_mode = True

if vegas_mode:
    Ptot[K > maxkhrec] = np.inf
    Pnlinsign[K > maxkhrec] = 0
    Pnlinsign[K < minkhrec] = 0

Pnlinsign_scipy = scipy.interpolate.interp1d(K, Pnlinsign, fill_value = 0., bounds_error = False)
Pidentity_scipy = scipy.interpolate.interp1d(K, Pnlinsign*0.+1.)

est = es.Estimator(K, Ptot, Plin)

#this object is using sympy to define the different modecoupling kernels

# F_g = 17/21
est.addF('g', 17./21.)
# F_s = (1/2) * (q_2/q_1 + q_1/q_2) * \vec{q}_1\cdot\vec{q}_2 / (q_1 q_2)
est.addF('s', 0.5*(est.q2/est.q1+est.q1/est.q2)*est.mu)
# F_t = (2/7) * [ (\vec{q}_1\cdot\vec{q}_2)^2 / (q_1^2 q_2^2) - 1/3 ]
est.addF('t', (2./7.)*est.mu**2.-1./3.)
# F_{11} = (1/2) * (1/M(q_1) + 1/M(q_2))
est.addF('c11', 0.5*(1./M(est.q1)+1./M(est.q2)))
# F_{01} = (1/2) * \vec{q}_1\cdot\vec{q}_2 / (q_1 q_2)
#            * ( 1/(q_1^2 M(q_2)) + 1/(q_2^2 M(q_1)) )
est.addF('c01', 0.5 * est.mu*est.q1*est.q2 \
                * (1./(est.q1**2.*M(est.q2))+1./(est.q2**2.*M(est.q1))) )
# F_{02} = 1 / (M(q_1) + M(q_2))
est.addF('c02', (1./(M(est.q1)*M(est.q2))))
# F_{\phi\phi} = M(|\vec{q}_1+\vec{q}_2|) / (M(q_1) M(q_2))
est.addF('phiphi', M(sp.sqrt(est.q1**2.+est.q2**2.+2*est.q1*est.q2*est.mu)) \
                    * (1./M(est.q1)) * (1./M(est.q2)) )


#Which modes are reconstructed
K_of_interest = np.arange(minkh, maxkh, 0.001)

#Now calculate different noise curves and store them inside the object
est.generateNs(K_of_interest, minkhrec, maxkhrec, vegas_mode = vegas_mode)

M = Mscipy(est.Krange)

values = np.array(list(est.keys))

listKeys = list(itertools.combinations_with_replacement(list(values), 2))

dic = {}

dic['K'] = est.Krange

Plin = np.interp(est.Krange, K, Plin)

for a, b in listKeys:
    dic['N'+a+b] = est.getN(a, b)
    dic['N'+b+a] = dic['N'+a+b]


new_bias = 1000 ##to be decided if I have to use sympy also here

for vv in variables_list:
    if 'N' not in vv:
        dic[vv] = globals()[vv]

with open(direc+data_dir+dic_name, 'wb') as handle:
    pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done')


'''
#Now calculate shot noise contributions to the bispectrum
Schematically speaking
int_q g_a*sum of terms
 g_a = N_a * f_a/(2*P*P)
'''

'''
Ngg = est.getN('g', 'g')

a = 'g'
mu_sign = 1.

shotfactor_zeroPpower = est.integrate_for_shot('g', K_of_interest, mu_sign,
    minkhrec, maxkhrec, Pidentity_scipy, Pidentity_scipy, vegas_mode = vegas_mode)
sh_bis_1 = (Ngg*shotfactor_zeroPpower)*shot**2.

sh_bis_3 = Pnlinsign_scipy(K_of_interest)*(shotfactor_zeroPpower*Ngg)*shot

shotfactor_onePpower = est.integrate_for_shot('g', K_of_interest, mu_sign,
    minkhrec, maxkhrec, Pnlinsign_scipy, Pidentity_scipy, vegas_mode = vegas_mode)

sh_bis_2 = (shotfactor_onePpower*Ngg)*shot

sh_bis = sh_bis_1+2*sh_bis_2+sh_bis_3
#This contribution goes to the cross spectrum between new field and original one


#Now calculate shot noise contributions to the trispectrum

shotfactor_zeroPpower_opposite = est.integrate_for_shot('g', K_of_interest,
    -mu_sign, minkhrec, maxkhrec, Pidentity_scipy, Pidentity_scipy, vegas_mode = vegas_mode)
shotfactor_onePpower_opposite = est.integrate_for_shot('g', K_of_interest,
    -mu_sign, minkhrec, maxkhrec, Pnlinsign_scipy, Pidentity_scipy, vegas_mode = vegas_mode)

shotfactor_twoPpower = est.integrate_for_shot('g', K_of_interest, mu_sign,
    minkhrec, maxkhrec, Pnlinsign_scipy, Pnlinsign_scipy, vegas_mode = vegas_mode)

shotfactor_double = est.double_integrate_for_shot(a, K_of_interest, mu_sign,
    minkhrec, maxkhrec, -mu_sign, minkhrec, maxkhrec, Pnlinsign_scipy, vegas_mode = vegas_mode)

sh_tris_1 = (shotfactor_zeroPpower*Ngg)**2.*shot**3

sh_tris_2 = (shotfactor_zeroPpower*Ngg)*((shotfactor_onePpower*Ngg))*shot**2.

sh_tris_3_a = (shotfactor_onePpower*Ngg)**2.*shot

sh_tris_3_b = (shotfactor_zeroPpower*Ngg)*(shotfactor_twoPpower*Ngg)*shot

sh_tris = sh_tris_1+4*sh_tris_2+4*sh_tris_3_a+2*sh_tris_3_b

#######DICTIONARTY OF STUFF###########

prefac = 1.

M = Mscipy(est.Krange)

term_labels = np.array(list(est.keys))

listKeys = list(itertools.combinations_with_replacement(list(term_labels), 2))

dic = {}

dic['K'] = est.Krange

for a, b in listKeys:
    dic['N'+a+b] = prefac**2.*est.getN(a, b)
    dic['N'+b+a] = dic['N'+a+b]

B = betaf
C = 4*deltac*((deltac/a1**2.)*(b20-2.*(a1+a2)*(bg-1.))-2.*(bg-1.))
A = (2./a1)*(deltac*(b20-2*(a1+a2)*(bg-1.))-a1**2.*(bg-1.))+2.*deltac*(bg-1.)


dic['minkh'] = minkh
dic['maxkh'] = maxkh
dic['minkhrec'] = minkhrec
dic['maxkhrec'] = maxkhrec
dic['z'] = z
dic['ngal'] = nhalo
#from the loop parts
dic['kphiphi'] = fnl*b10
dic['kb01'] = fnl*b01
dic['kb11'] = fnl*b11
dic['kb02'] = fnl**2.*b02

dic['dfnlkphiphi'] = b10

dic['bg'] = b10
dic['ks'] = cs
dic['kt'] = ct
dic['kg'] = cg
dic['b20'] = b20
dic['bs2'] = bs2

dic['fnl'] = fnl
dic['bfnllargescales'] = betaf*fnl/M
dic['derbfnllargescales'] = betaf/M
dic['betaf'] = betaf
dic['M'] = M

Cgg = np.interp(est.Krange, K, Ptot)
PL = np.interp(est.Krange, K, Plin)
# dfnlCgg = 2*(b10(betaf*fnl*D(z))/M)*(betaf*D(z)/M)*PL
dfnlCgg = 2*(b10+(betaf*fnl/M))*(betaf/M)*PL

terms = []
for a in term_labels:
    terms += [dic['k'+a]*dic['Ngg']/dic['Ng'+a]]
partial = prefac*b10*sum(terms)

Cnn = (partial)**2.*PL+dic['Ngg']
derpartialfnl = prefac * b10 * \
    ( (b10*dic['Ngg']/dic['Ngphiphi']) + b11*(dic['Ngg']/dic['Ngb11']) \
        + b01*(dic['Ngg']/dic['Ngb01']) + 2*fnl*b02*(dic['Ngg']/dic['Ngb02'])
        )
dfnlCnn = 2*partial*derpartialfnl*PL

# Cgn = (b10+(betaf*fnl*D(z))/M)*partial*PL
# dfnlCgn = (b10+(betaf*fnl*D(z))/M)*derpartialfnl*PL+(betaf*D(z)/M)*partial*PL
Cgn = (b10+(betaf*fnl/M))*partial*PL
dfnlCgn = (b10+(betaf*fnl/M))*derpartialfnl*PL+(betaf/M)*partial*PL

dic['Cgg'] = Cgg
dic['Cnn'] = Cnn
dic['Cgn'] = Cgn
dic['dfnlCgg'] = dfnlCgg
dic['dfnlCgn'] = dfnlCgn
dic['dfnlCnn'] = dfnlCnn
dic['PL'] = PL
dic['shotnoise'] = shot
# dic['shot_noise_bispectrum'] = shot_noise_bispectrum
# dic['shot_noise_trispectrum'] = shot_noise_trispectrum
dic['term_labels'] = term_labels
# TODO: fix line below so that we actually save the values we want
dic['values'] = values


with open(direc+'/data_dir/spectra.pickle', 'wb') as handle:
    pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done')
'''
