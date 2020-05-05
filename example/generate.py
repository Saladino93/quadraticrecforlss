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

tck = scipy.interpolate.splrep(kM, Mval)

def M2(k):
    return scipy.interpolate.splev(k, tck)

M = implemented_function('M', M2)    

# Define min and max k values, making sure M is defined over the full k range
minK = np.maximum(kM.min(), K.min())
maxK = np.minimum(kM.max(), K.max())

# Select k and power spectrum values within range defined above
select = (K>minK)&(K<maxK)
K = K[select]
Klin = Klin[select]
Plin = Plin[select]
Pnlin = Pnlin[select]


#################################
# Set some parameter values
#################################

# Fetch some values from the input dict


minkh = values['analysis_config']['mink_analysis'] 
maxkh = values['analysis_config']['maxk_analysis']

minkhrec = values['analysis_config']['mink_reconstruction']
maxkhrec = values['analysis_config']['maxk_reconstruction']

specific_combs = values['analysis_config']['specific_combs']

if specific_combs == '':
   specific_combs = None
else:
   a = specific_combs[0]
   b = specific_combs[1]

   l = []

   for i in a:
      for j in b:
            temp = tuple([i, j])
            temp_ = tuple([j, i])
            condition = (temp in l) or (temp_ in l)
            if not condition:
               l += [temp]
   
   specific_combs = l

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

f = values['survey_config']['geometry']['f']

f = float(f)

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

mu = np.linspace(-1, 1, len(K))

M_K = Mscipy(K)
inv_M_mesh, mu_mesh = np.meshgrid(1/M_K, mu)

b_tot = b10+(betaf*fnl)*inv_M_mesh+f*mu_mesh**2.

Ptot = (b_tot)**2.*Pnlin+shot

Pnlinsign = (b_tot)**2.*Pnlin


''' CREATE ESTIMATOR OBJECT FOR NOISE CALCULATIONS '''

vegas_mode = True

est = es.Estimator(minkhrec, maxkhrec, K, mu, Ptot, Plin, Pnlinsign, b_tot, nhalo)

#this object is using sympy to define the different modecoupling kernels

g = b10+21/17*b20
s = b10
t = b10+7/2*bs2
phiphi = fnl*b10
c01 = fnl*2*deltac*(b10-1)
c11 = fnl*(2./a1)*(deltac*(b20-2*(a1+a2)*(b10-1.))-a1**2.*(b10-1.))+2.*fnl*deltac*(b10-1.)
c02 = fnl**2*4*deltac*((deltac/a1**2.)*(b20-2.*(a1+a2)*(b10-1.))-2.*(b10-1.))


# F_g = 17/21
est.addF('g', 17./21., ca = g)
# F_s = (1/2) * (q_2/q_1 + q_1/q_2) * \vec{q}_1\cdot\vec{q}_2 / (q_1 q_2)
est.addF('s', 0.5*(est.q2/est.q1+est.q1/est.q2)*est.mu, ca = s)
# F_t = (2/7) * [ (\vec{q}_1\cdot\vec{q}_2)^2 / (q_1^2 q_2^2) - 1/3 ]
est.addF('t', (2./7.)*est.mu**2.-1./3., ca = t)
# F_{11} = (1/2) * (1/M(q_1) + 1/M(q_2))
est.addF('c11', 0.5*(1./M(est.q1)+1./M(est.q2)), ca = c11)
# F_{01} = (1/2) * \vec{q}_1\cdot\vec{q}_2 / (q_1 q_2)
#            * ( 1/(q_1^2 M(q_2)) + 1/(q_2^2 M(q_1)) )
est.addF('c01', 0.5 * est.mu*est.q1*est.q2 \
                * (1./(est.q1**2.*M(est.q2))+1./(est.q2**2.*M(est.q1))), ca = c01)
# F_{02} = 1 / (M(q_1) + M(q_2))
est.addF('c02', (1./(M(est.q1)*M(est.q2))), ca = c02)
# F_{\phi\phi} = M(|\vec{q}_1+\vec{q}_2|) / (M(q_1) M(q_2))
est.addF('phiphi', M(sp.sqrt(est.q1**2.+est.q2**2.+2*est.q1*est.q2*est.mu)) \
                    * (1./M(est.q1)) * (1./M(est.q2)), ca = phiphi)


#Which modes are reconstructed
K_of_interest = np.arange(minkh, maxkh, 0.02)
mu_of_interest = np.linspace(-1, 1, len(K_of_interest))

#Now calculate different noise curves and store them inside the object
est.generateNs(K_of_interest, mu_of_interest, minkhrec, maxkhrec, specific_combs, vegas_mode = vegas_mode)


M = Mscipy(est.Krange)

values = np.array(list(est.keys))

if specific_combs is None:
    listKeys = list(itertools.combinations_with_replacement(list(values), 2))
else:
    listKeys = specific_combs


dic = {}

dic['K'] = est.Krange

Plin = np.interp(est.Krange, Klin, Plin)

shot = dic['K']*0.+shot

for a, b in listKeys:
    dic['N'+a+b] = est.getN(a, b)
    dic['N'+b+a] = dic['N'+a+b]

new_bias = 0.
for k in est.keys:
    new_bias += b10*est.getN('g', 'g')*est.getN('g', k)**-1.*est.c_a[k]

sh_bis = est.get_bispectrum_shot_noise('g', K = est.Krange, minkhrec, maxkhrec)
import time
s = time.time()
sh_tris = est.get_trispectrum_shot_noise('g', K = est.Krange, minq = minkhrec, maxq = maxkhrec)
delta = time.time()-s
print(f'Total time for trispectrum {delta}')



'''
#Now calculate shot noise contributions to the bispectrum
Schematically speaking
int_q g_a*sum of terms
 g_a = N_a * f_a/(2*P*P)
'''



index_max = np.where(K<maxkhrec)[0][-1]
index_min = np.where(K>minkhrec)[0][0]

Ptot = Ptot[:, index_min:index_max] #slice so that scipy interp takes care of filling np.inf
Plin_slice = Plin[index_min:index_max]
Pnlinsign = Pnlinsign[:, index_min:index_max]
K = K[index_min:index_max]

Pnlinsign_scipy = scipy.interpolate.interp1d(K, Pnlinsign, fill_value = 0., bounds_error = False)
Pidentity_scipy = scipy.interpolate.interp1d(K, Pnlinsign*0.+1., fill_value = 0., bounds_error = False)

Pnlinsign_scipy2dd = scipy.interpolate.interp2d(K, mu, Pnlinsign, fill_value = 0., bounds_error = False)
Pidentity_scipy2dd = scipy.interpolate.interp2d(K, mu, Pnlinsign*0.+1., fill_value = 0., bounds_error = False)

Ptot2d = scipy.interpolate.interp2d(K, mu, Ptot, fill_value = 0., bounds_error = False)

min_x1, max_x1, min_x2, max_x2 = K.min(), K.max(), mu.min(), mu.max()
fill_value = 0.
Pnlinsign_scipy2d = lambda q, mu: es.vectorize_2dinterp(Pnlinsign_scipy2dd, q, mu, min_x1, max_x1, fill_value = fill_value)
Pidentity_scipy2d = lambda q, mu: es.vectorize_2dinterp(Pidentity_scipy2dd, q, mu, min_x1, max_x1, fill_value = fill_value)

print('Getting bispectrum shot noise contribution')

Ngg = est.getN('g', 'g')

a = 'g'
mu_sign = 1.

shotfactor_zeroPpower = est.integrate_for_shot('g', K_of_interest, mu_sign, minkhrec, maxkhrec, Pidentity_scipy2d, Pidentity_scipy2d, vegas_mode = vegas_mode)
sh_bis_1 = (Ngg*shotfactor_zeroPpower)*shot**2.

sh_bis_3 = (shotfactor_zeroPpower*Ngg)*Pnlinsign_scipy2d(K_of_interest, mu_of_interest)*shot

shotfactor_onePpower = est.integrate_for_shot('g', K_of_interest, mu_sign, minkhrec, maxkhrec, Pnlinsign_scipy2d, Pidentity_scipy2d, vegas_mode = vegas_mode)

sh_bis_2 = (shotfactor_onePpower*Ngg)*shot

sh_bis = sh_bis_1+2*sh_bis_2+sh_bis_3 #This contribution goes to the cross spectrum between new field and original one

print('Getting trispectrum shot noise contribution')

#Now calculate shot noise contributions to the trispectrum

shotfactor_zeroPpower_opposite = est.integrate_for_shot('g', K_of_interest, -mu_sign, minkhrec, maxkhrec, Pidentity_scipy2d, Pidentity_scipy2d, vegas_mode = vegas_mode)
shotfactor_onePpower_opposite = est.integrate_for_shot('g', K_of_interest, -mu_sign, minkhrec, maxkhrec, Pnlinsign_scipy2d, Pidentity_scipy2d, vegas_mode = vegas_mode)

shotfactor_twoPpower = est.integrate_for_shot('g', K_of_interest, mu_sign, minkhrec, maxkhrec, Pnlinsign_scipy2d, Pnlinsign_scipy2d, vegas_mode = vegas_mode)


shotfactor_double = est.double_integrate_for_shot(a, K_of_interest, mu_sign, minkhrec, maxkhrec, -mu_sign, minkhrec, maxkhrec, Pnlinsign_scipy2d, vegas_mode = vegas_mode)

sh_tris_1 = (shotfactor_zeroPpower*Ngg)**2.*shot**3

sh_tris_2 = (shotfactor_zeroPpower*Ngg)*((shotfactor_onePpower*Ngg))*shot**2.

sh_tris_3_a = (shotfactor_onePpower*Ngg)**2.*shot 

sh_tris_3_b = (shotfactor_zeroPpower*Ngg)*(shotfactor_twoPpower*Ngg)*shot

sh_tris_4_a = shotfactor_double*Ngg**2.*shot**2.
sh_tris_4_b = shotfactor_zeroPpower**2*Ngg**2.*shot**2.*Pnlinsign_scipy2d(K_of_interest, mu_of_interest)

sh_tris = sh_tris_1+4*sh_tris_2+4*sh_tris_3_a+2*sh_tris_3_b+2*sh_tris_4_a+sh_tris_4_b


print('sh_tris_1 n3', sh_tris_1)
print(sh_tris_2)
print(sh_tris_3_a)
print(sh_tris_3_b)
print(sh_tris_4_a)
print(sh_tris_4_b)
print(sh_tris)


print('')
print(est.trispectrum_shot)
print(sh_tris)

print(est.trispectrum_shot[0, :]/est.getN('g', 'g'))


for vv in variables_list:
    if 'N' not in vv:
        dic[vv] = globals()[vv]

M_mesh, mu_mesh = np.meshgrid(Mscipy(K_of_interest), mu_of_interest)

dic['mu'] = mu_mesh#mu_of_interest
dic['M'] = M_mesh

dic['Ptot'] = Ptot

with open(direc+data_dir+dic_name, 'wb') as handle:
    pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done')
