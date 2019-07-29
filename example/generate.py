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

Mtransferfile = 'M.txt'

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

betaf = 2.*deltac*(bg-1.)

shot = 1/nbar
Ptot = (bg+(betaf*fnl*D(z))/Mscipy(K))**2.*Pnlin+shot
Pnlinsign = Pnlin#(bg+(betaf*fnl*D(z))/Mscipy(K))**2.*Pnlin

cg = bg+b20/2.*(7./5.)+(7./5.)*(2./21.)*(bg-1)
cs = bg*1
ct = bg*1#bg*ct-(2./7.)*(bg-1.)


############ Begin Calculations

K_of_interest = np.arange(minkh, maxkh, 0.001)

est = es.Estimator(K, Ptot, Plin)

est.addF('g', 5./7.) 
est.addF('s', 0.5*(est.q2/est.q1+est.q1/est.q2)*est.mu)
est.addF('t', (2./7.)*est.mu**2.)
est.addF('b11', 0.5*(1./M(est.q1)+1./M(est.q2)))
est.addF('b01', 0.5*est.mu*est.q1*est.q2*(1./(est.q1**2.*M(est.q2))+1./(est.q2**2.*M(est.q1))))
est.addF('b02', 0.5*(1./(M(est.q1)*M(est.q2))))
est.addF('phiphi', M(sp.sqrt(est.q1**2.+est.q2**2.+2*est.q1*est.q2*est.mu))*(1./M(est.q1))*(1./M(est.q2)))

est.generateNs(K_of_interest, minkhrec, maxkhrec)



#######DICTIONARTY OF STUFF

M = Mscipy(est.Krange)

values = np.array(['g', 's', 't', 'b11', 'b01', 'b02', 'phiphi'])
listKeys = list(itertools.combinations_with_replacement(list(values), 2))

dic = {}

dic['K'] = est.Krange

for a, b in listKeys:
    dic['N'+a+b] = est.getN(a, b)
    dic['N'+b+a] = dic['N'+a+b]

betaf = 2.*deltac*(bg-1.)
B = betaf
C = 4*deltac*((deltac/a1**2.)*(b20-2.*(a1+a2)*(bg-1.))-2.*(bg-1.))
A = (2./a1**2.)*(deltac*(b20-2*(a1+a2)*(bg-1.))-a1**2.*(bg-1.))+2.*deltac*(bg-1.)

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
partial = bg*sum(terms)

Cnn = (partial)**2.*PL+dic['Ngg']
derpartialfnl = (bg*dic['Ngg']/dic['Ngphiphi'])+A*(dic['Ngg']/dic['Ngb11'])+B*(dic['Ngg']/dic['Ngb01'])+2*fnl*C*(dic['Ngg']/dic['Ngb02'])
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

dic['values'] = values

with open(direc+'/data_dir/spectra.pickle', 'wb') as handle:
    pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done')
