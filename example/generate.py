import sys

from quadforlss import estimator as es

import numpy as np

import pickle

import sympy as sp
from sympy.utilities.lambdify  import implemented_function

import scipy.interpolate

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

Mhalo = values['Mhalo']

b20 = values['b20']
bs2 = values['bs2']

betaf = 2.*deltac*(bg-1.)

shot = 1/nbar
Ptot = (bg+(betaf*fnl)/Mscipy(K))**2.*Pnlin+shot
Pnlinsign = (bg+(betaf*fnl*D(z))/Mscipy(K))**2.*Pnlin

Pnlinsign_scipy = scipy.interpolate.interp1d(K, Pnlinsign)

Pidentity_scipy = scipy.interpolate.interp1d(K, Pnlinsign*0.+1.)

cg = bg+b20/2.*(7./5.)
cs = bg*1
ct = bg+7/2.*bs2

############ Begin Calculations

vegas_mode = True

if vegas_mode:
    Ptot[K > maxkhrec] = np.inf
    Pnlinsign[K > maxkhrec] = 0
    Pnlinsign[K > minkhrec] = 0

Pnlinsign_scipy = scipy.interpolate.interp1d(K, Pnlinsign, fill_value = 0., bounds_error = False)

K_of_interest = np.arange(minkh, maxkh, 0.001)

est = es.Estimator(K, Ptot, Plin)

est.addF('g', 5./7.) 
est.addF('s', 0.5*(est.q2/est.q1+est.q1/est.q2)*est.mu)
est.addF('t', (2./7.)*est.mu**2.)
est.addF('b11', 0.5*(1./M(est.q1)+1./M(est.q2)))
est.addF('b01', 0.5*est.mu*est.q1*est.q2*(1./(est.q1**2.*M(est.q2))+1./(est.q2**2.*M(est.q1))))
est.addF('b02', (1./(M(est.q1)*M(est.q2))))
est.addF('phiphi', M(sp.sqrt(est.q1**2.+est.q2**2.+2*est.q1*est.q2*est.mu))*(1./M(est.q1))*(1./M(est.q2)))

a = 'g'
mu_sign = 1.

est.generateNs(K_of_interest, minkhrec, maxkhrec, vegas_mode = vegas_mode)

Ngg = est.getN('g', 'g')

mu_sign = 1.

shotfactor = est.integrate_for_shot('g', K_of_interest, mu_sign, minkhrec, maxkhrec, Pidentity_scipy, Pidentity_scipy, vegas_mode = vegas_mode)

shotfactor_id_id = shotfactor

sh_bis_1 = shotfactor*Ngg*shot**2.

sh_tris_1 = (shotfactor*Ngg)**2.*shot**3

sh_bis_3 = Pnlinsign_scipy(K_of_interest)*shotfactor*Ngg*shot

g_int = shotfactor*Ngg

shotfactor = est.integrate_for_shot('g', K_of_interest, mu_sign, minkhrec, maxkhrec, Pnlinsign_scipy, Pidentity_scipy, vegas_mode = vegas_mode)

sh_bis_2 = shotfactor*Ngg*shot

sh_tris_2 = shotfactor*Ngg*g_int*shot**2.

g_int_p = shotfactor*Ngg

sh_tris_3_a = g_int_p**2.*shot

shot_noise_bispectrum = sh_bis_1+2*sh_bis_2+sh_bis_3

shotfactor = est.integrate_for_shot('g', K_of_interest, mu_sign, minkhrec, maxkhrec, Pnlinsign_scipy, Pnlinsign_scipy, vegas_mode = vegas_mode)

sh_tris_3_b = shotfactor*Ngg*shot*g_int

shotfactor = est.double_integrate_for_shot(a, K_of_interest, mu_sign, minkhrec, maxkhrec, -mu_sign, minkhrec, maxkhrec, Pnlinsign_scipy, vegas_mode = vegas_mode)
sh_tris_4_a = shotfactor*Ngg**2.*shot**2.
sh_tris_4_b = shotfactor_id_id**2*Ngg**2.*shot**2.*Pnlinsign_scipy(K_of_interest)

shot_noise_trispectrum = sh_tris_1+4*sh_tris_2+4*sh_tris_3_a+2*sh_tris_3_b+2*sh_tris_4_a+sh_tris_4_b

'''
shotfactor = est.integrate_for_shot('g', K_of_interest, mu_sign, minkhrec, maxkhrec, Pnlinsign_scipy, Pidentity_scipy, vegas_mode = vegas_mode)
print(shotfactor)
mu_sign = -1.
shotfactor = est.integrate_for_shot('g', K_of_interest, mu_sign, minkhrec, maxkhrec, Pnlinsign_scipy, Pidentity_scipy, vegas_mode = vegas_mode)
print(shotfactor)
'''

#######DICTIONARTY OF STUFF

prefac = 1.

M = Mscipy(est.Krange)

values = np.array(list(est.keys))

listKeys = list(itertools.combinations_with_replacement(list(values), 2))

dic = {}

dic['Mhalo'] = Mhalo

dic['K'] = est.Krange

for a, b in listKeys:
    dic['N'+a+b] = prefac**2.*est.getN(a, b)
    dic['N'+b+a] = dic['N'+a+b]

betaf = 2.*deltac*(bg-1.)
B = betaf
C = 4*deltac*((deltac/a1**2.)*(b20-2.*(a1+a2)*(bg-1.))-2.*(bg-1.))
A = (2./a1)*(deltac*(b20-2*(a1+a2)*(bg-1.))-a1**2.*(bg-1.))+2.*deltac*(bg-1.)

dic['minkh'] = minkh
dic['maxkh'] = maxkh
dic['minkhrec'] = minkhrec
dic['maxkhrec'] = maxkhrec
dic['z'] = z
dic['ngal'] = nbar
#from the loop parts
dic['kphiphi'] = fnl*bg
dic['kb01'] = fnl*B
dic['kb11'] = fnl*A
dic['kb02'] = fnl**2.*C

dic['dfnlkphiphi'] = bg

dic['bg'] = bg
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
dfnlCgg = 2*(bg+(betaf*fnl*D(z))/M)*(betaf*D(z)/M)*PL

terms = []
for a in values:
    terms += [dic['k'+a]*dic['Ngg']/dic['Ng'+a]]
partial = prefac*bg*sum(terms)

Cnn = (partial)**2.*PL+dic['Ngg']
derpartialfnl = prefac*bg*((bg*dic['Ngg']/dic['Ngphiphi'])+A*(dic['Ngg']/dic['Ngb11'])+B*(dic['Ngg']/dic['Ngb01'])+2*fnl*C*(dic['Ngg']/dic['Ngb02']))
dfnlCnn = 2*partial*derpartialfnl*PL

Cgn = (bg+(betaf*fnl*D(z))/M)*partial*PL
dfnlCgn = (bg+(betaf*fnl*D(z))/M)*derpartialfnl*PL+(betaf*D(z)/M)*partial*PL

dic['Cgg'] = Cgg
dic['Cnn'] = Cnn
dic['Cgn'] = Cgn
dic['dfnlCgg'] = dfnlCgg
dic['dfnlCgn'] = dfnlCgn
dic['dfnlCnn'] = dfnlCnn
dic['PL'] = PL
dic['shotnoise'] = shot
dic['shot_noise_bispectrum'] = shot_noise_bispectrum
dic['shot_noise_trispectrum'] = shot_noise_trispectrum
dic['values'] = values


with open(direc+'/data_dir/spectra.pickle', 'wb') as handle:
    pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done')
