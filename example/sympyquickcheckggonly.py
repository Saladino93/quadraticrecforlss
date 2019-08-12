import sys

from quadforlss import forecasting as fore

from quadforlss import estimator

import opentext

import numpy as np

from matplotlib import pyplot as plt

import scipy

import sympy as sp

import pickle

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


with open(direc+'/data_dir/spectra.pickle', 'rb') as handle:
    dic = pickle.load(handle, encoding='latin1')

nonlinpowerfile = direc+'/nonlinear_power.txt'
linpowerfile = direc+'/linear_power.txt'
Mtransferfile = direc+'/M.txt'

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
mu = np.arange(-1.1, 1, 2.1/len(K))

bg = 1.6
ngal = 1e-2#2*1e-4#1e-2
Nshot = 1/ngal+0.*K
fnlfid = 0.
deltac = 1.42
betaf = 2*deltac*(bg-1.)
h = 0.67
Omega0_m = 0.3089037414433296
f0 = Omega0_m**0.6/bg

fac = betaf/Mval

Ptot = (bg+fac*fnlfid+f0*mu**2.)**2.*Pnlin+Nshot
dPtot = 2*(bg+fac*fnlfid+f0*mu**2.)*fac*Pnlin

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

dernbar_P_s = sp.diff(P_total, nbar)
dernbar_P = sp.lambdify([b, fnl, nbar, Pnl, fnlfunc], dernbar_P_s, 'numpy')

derbP = derb_P(bg, fnlfid, ngal, Pnlin, 1./Mval)
derbPgg = derbP
derfnlP = derfnl_P(bg, fnlfid, ngal, Pnlin, 1./Mval)
dernbarP = dernbar_P(bg, fnlfid, ngal, Pnlin, 1./Mval)

derbP2 = 2*Pnlin*(bg+fnlfid*2.*deltac*(bg-1.)/Mval)*(1+2*deltac/Mval)
derfnlP2 = 2*Pnlin*(bg+fnlfid*2.*deltac*(bg-1.)/Mval)*2.*deltac*(bg-1.)/Mval

fbb = fab(Pgg, derbP, derbP)
ffnlfnl = fab(Pgg, derfnlP, derfnlP)
ffnlb = fab(Pgg, derfnlP, derbP)
fnbarnbar = fab(Pgg, dernbarP, dernbarP)
ffnlnbar = fab(Pgg, derfnlP, dernbarP)
fbnbar = fab(Pgg, derbP, dernbarP)

kmin_ksz = 0.00066#768356195
kmax_ksz = 0.11

sel = np.where((K > kmin_ksz) & (K < kmax_ksz))
K = K[sel]
mu = mu[sel]
kmin = np.min(K)
kmax = np.max(K)
V = (np.pi)**3./kmin**3.
Ks = np.arange(kmin, kmax/1.5, 0.001)
print('kmin, ', kmin, ' V,(Gpc^3) ', V/1e9)
fbb = fbb[sel]
ffnlfnl = ffnlfnl[sel]
ffnlb = ffnlb[sel]
fnbarnbar = fnbarnbar[sel]
ffnlnbar = ffnlnbar[sel]
fbnbar = fbnbar[sel]

func_fbb = scipy.interpolate.interp1d(K, fbb)
func_ffnlfnl = scipy.interpolate.interp1d(K, ffnlfnl)
func_ffnlb = scipy.interpolate.interp1d(K, ffnlb)
func_fnbarnbar = scipy.interpolate.interp1d(K, fnbarnbar)
func_ffnlnbar = scipy.interpolate.interp1d(K, ffnlnbar)
func_fbnbar = scipy.interpolate.interp1d(K, fbnbar)



'''
f = Omega0_m**0.6
bias = bg+fnlfid*2*(bg-1.)*deltac/Mval+(f/bg)*mu**2.

derb_bias = 1+fnlfid*2*deltac/Mval-(f/bg**2.)*mu**2. 
derfnl_bias = 2*(bg-1.)*deltac/Mval+K*0.
dermu_bias = (1/bg)*mu**2.

bias = scipy.interpolate.interp1d(mu, bias)
derb_bias = scipy.interpolate.interp1d(mu, derb_bias)
derfnl_bias = scipy.interpolate.interp1d(mu, derfnl_bias)
dermu_bias = scipy.interpolate.interp1d(mu, dermu_bias)

Pnlin = scipy.interpolate.interp1d(K, Pnlin)

def Ptot_func(K, mu):
    return bias(mu)**2.*Pnlin(K)+1./ngal

def derb_Ptot_func(K, mu):
    return 2*bias(mu)*derb_bias(mu)*Pnlin(K)

def derfnl_Ptot_func(K, mu):
    return 2*bias(mu)*derfnl_bias(mu)*Pnlin(K)

def derf_Ptot_func(K, mu):
    return 2*bias(mu)*dermu_bias(mu)*Pnlin(K)

def Fisher(a, b):

    def Fbb(mu, K):
        result = 0.5*(derb_Ptot_func(K, mu)/Ptot_func(K, mu))*(derb_Ptot_func(K, mu)/Ptot_func(K, mu))
        return result

    def Fbfnl(mu, K):
        result = 0.5*(derb_Ptot_func(K, mu)/Ptot_func(K, mu))*(derfnl_Ptot_func(K, mu)/Ptot_func(K, mu))
        return result

    def Fbf(mu, K):
        result = 0.5*(derb_Ptot_func(K, mu)/Ptot_func(K, mu))*(derf_Ptot_func(K, mu)/Ptot_func(K, mu))
        return result

    def Ffnlfnl(mu, K):
        result = 0.5*(derfnl_Ptot_func(K, mu)/Ptot_func(K, mu))*(derfnl_Ptot_func(K, mu)/Ptot_func(K, mu))
        return result

    def Ffnlf(mu, K):
        result = 0.5*(derfnl_Ptot_func(K, mu)/Ptot_func(K, mu))*(derf_Ptot_func(K, mu)/Ptot_func(K, mu))
        return result

    def Fff(mu, K):
        result = 0.5*(derf_Ptot_func(K, mu)/Ptot_func(K, mu))*(derf_Ptot_func(K, mu)/Ptot_func(K, mu))
        return result

    if (a == 'b') and (b == 'b'):
        return Fbb
    elif (a == 'b') and (b == 'fnl'):
        return Fbfnl
    elif (a == 'b') and (b == 'f'):
        return Fbf
    elif (a == 'fnl') and (b == 'fnl'):
        return Ffnlfnl
    elif (a == 'fnl') and (b == 'f'):
        return Ffnlf
    elif (a == 'f') and (b == 'f'):
        return Fff
'''

def Fisher(a, b):
    if (a == 'b') and (b == 'b'):
        return func_fbb
    elif (a == 'b') and (b == 'fnl'):
        return func_ffnlb
    elif (a == 'fnl') and (b == 'fnl'):
        return func_ffnlfnl


kmin_ksz = 0.00066768356195
kmax_ksz = 0.11

sel = np.where((K > kmin_ksz) & (K < kmax_ksz))
K = K[sel]
#mu = mu[sel]
kmin = np.min(K)
kmax = np.max(K)
V = (np.pi)**3./kmin**3.
Ks = np.arange(kmin, kmax/1.5, 0.001)
print('kmin, ', kmin, ' V,(Gpc^3) ', V/1e9)

mat = np.zeros((2, 2))
#mat = np.zeros((3, 3))

fint = []

errfnlmarg = []
print('Do integration')
for k in Ks: #Fisher('b', 'b')
    mat[0, 0] = scipy.integrate.quad(lambda x: Fisher('b', 'b')(x)*x**2., kmin, kmax)[0]   
    mat[1, 1] = scipy.integrate.quad(lambda x: Fisher('fnl', 'fnl')(x)*x**2., kmin, kmax)[0]
    mat[0, 1] = scipy.integrate.quad(lambda x: Fisher('b', 'fnl')(x)*x**2., kmin, kmax)[0]
    mat[1, 0] = mat[0, 1]
    #mat[2, 2] = scipy.integrate.dblquad(Fisher('fnl', 'fnl'), kmin, kmax, lambda x: -1., lambda x :1.)[0]
    #mat[1, 2] = scipy.integrate.dblquad(Fisher('fnl', 'f'), kmin, kmax, lambda x: -1., lambda x :1.)[0]
    #mat[2, 1] = mat[1, 2]
    #mat[0, 2] = scipy.integrate.dblquad(Fisher('f', 'f'), kmin, kmax, lambda x: -1., lambda x :1.)[0]
    #mat[2, 0] = mat[0, 2]
    mat *= V/(2*np.pi**2.)
    invmat = np.linalg.inv(mat)
    el = np.sqrt(invmat[1, 1])
    errfnlmarg += [el]

errfnlmarg = np.array(errfnlmarg)

func_f = Fisher('fnl', 'fnl')


fint = []

for k in Ks:
    mat = scipy.integrate.quad(lambda x: func_f(x)*x**2., k, kmax)[0]
    mat *= V/(2*np.pi**2.)
    fint += [mat]
    

fint = np.array(fint)
invf = fint**-0.5

invf_gonly = invf.copy()
errfnlmarg_gonly = errfnlmarg.copy()
Ks_gonly = Ks.copy()

plt.xlabel('$K_{min}$ (Mpc$^{-1}$)')
plt.ylabel('$ Integrated \sigma(f_{nl}) $')
plt.loglog(Ks, invf, label = 'Galaxies Integrated $\sigma(fnl = $'+str(fnlfid)+') for $V = $'+'{:.2E}'.format(V/1e9)+'$Gpc^3$')
plt.loglog(Ks, errfnlmarg, label = 'Marginalized')
#plt.loglog(K, f**-0.5, label = 'PerMode')
plt.legend(loc = 'best', prop = {'size': 6})
plt.rc('grid', linestyle = "-", color = 'black')
plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-')
plt.minorticks_on()
plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2)
plt.ylim(bottom = 5, top = 300)
plt.savefig(output+'fnlplot.png', dpi = 300)
plt.close()

############# COMBINED PART

a1 = values['a1']
a2 = values['a2']

z = values['z']

b, b2, fnl, nbar, Pnl, fnlfunc = sp.symbols('b b2 fnl nbar Pnl fnlfunc')

cg = b+7./5.*(b2/2+2./21.*(b-1))
cs = b
ct = b
cphiphi = b
c01 = 2*deltac*(b-1)
c11 = (2./a1**2.)*(deltac*(b2-2*(a1+a2)*(b-1.))-a1**2.*(b-1.))+2.*deltac*(b-1.)
c02 = 4*deltac*((deltac/a1**2.)*(b2-2.*(a1+a2)*(b-1.))-2.*(b-1.))

Cgg = dic['Cgg']
Cnn = dic['Cnn']
Cgn = dic['Cgn']

bg = dic['bg'] #gal bias
b20 = dic['b20']

Ngg = dic['Ngg']
Ngs = dic['Ngs']
Ngt = dic['Ngt']
Ngb01 = dic['Ngb01']
Ngb02 = dic['Ngb02']
Ngb11 = dic['Ngb11']
Ngphiphi = dic['Ngphiphi']

KK = dic['K']

Mval = dic['M']

fnlfid = dic['fnl']

fs = Ngg/Ngs
ft = Ngg/Ngt
f01 = Ngg/Ngb01
f02 = Ngg/Ngb02
f11 = Ngg/Ngb11
fphiphi = Ngg/Ngphiphi

biasnew = b*(cg+cs*fs+ct*ft+(c01*f01+c02*f02+c11*f11+cphiphi*fphiphi)*fnl+c02*f02*fnl**2.)

der_b_biasnew_s = sp.diff(biasnew, b)
der_b_biasnew = sp.lambdify([b, b2, fnl], der_b_biasnew_s, 'numpy')
derb_biasnew = der_b_biasnew(bg, b20, fnlfid)

der_b2_biasnew_s = sp.diff(biasnew, b2)
der_b2_biasnew = sp.lambdify([b, b2, fnl], der_b2_biasnew_s, 'numpy')
derb2_biasnew = der_b2_biasnew(bg, b20, fnlfid)

der_fnl_biasnew_s = sp.diff(biasnew, fnl)
der_fnl_biasnew = sp.lambdify([b, b2, fnl], der_fnl_biasnew_s, 'numpy')
derfnl_biasnew = der_fnl_biasnew(bg, b20, fnlfid)

bias_numpy = sp.lambdify([b, b2, fnl], biasnew, 'numpy') 
biasnew = bias_numpy(bg, b20, fnlfid)

biasnew = np.array(biasnew)
derb_biasnew = np.array(derb_biasnew)
derb2_biasnew = np.array(derb2_biasnew)
derfnl_biasnew = np.array(derfnl_biasnew)

function = scipy.interpolate.interp1d(kM, Pnlin)
Pnlin = function(KK)

derb_Pn = 2*biasnew*derb_biasnew*Pnlin
derb2_Pn = 2*biasnew*derb2_biasnew*Pnlin
derfnl_Pn = 2*biasnew*derfnl_biasnew*Pnlin

invM = 1./Mval
func = 2*(b-1.)*(deltac*invM)
biasg = b+fnl*func

der_fnl_biasg_s = sp.diff(biasg, fnl)
der_fnl_biasg = sp.lambdify([b, b2, fnl], der_fnl_biasg_s, 'numpy')
derfnl_biasg = der_fnl_biasg(bg, b20, fnlfid)

der_b_biasg_s = sp.diff(biasg, b)
der_b_biasg = sp.lambdify([b, b2, fnl], der_b_biasg_s, 'numpy')
derb_biasg = der_b_biasg(bg, b20, fnlfid)

der_b2_biasg_s = sp.diff(biasg, b2)
der_b2_biasg = sp.lambdify([b, b2, fnl], der_b2_biasg_s, 'numpy')
derb2_biasg = der_b2_biasg(bg, b20, fnlfid)

biasg = sp.lambdify([b, b2, fnl], biasg, 'numpy')
biasg = biasg(bg, b20, fnlfid)

derb_biasg = np.array(derb_biasg)
derb2_biasg = np.array(derb2_biasg)
derfnl_biasg = np.array(derfnl_biasg)
biasg = np.array(biasg)

derb_Pg = 2*biasg*derb_biasg*Pnlin
derb2_Pg = 2*biasg*derb2_biasg*Pnlin
derfnl_Pg = 2*biasg*derfnl_biasg*Pnlin

derb_Pgn = (biasg*derb_biasnew+derb_biasg*biasnew)*Pnlin
derb2_Pgn = (biasg*derb2_biasnew+derb2_biasg*biasnew)*Pnlin
derfnl_Pgn = (biasg*derfnl_biasnew+derfnl_biasg*biasnew)*Pnlin


values = dic['values'].astype(str)
terms = []
for a in values:
    terms += [dic['k'+a]*dic['Ngg']/dic['Ng'+a]]
partial = bg*sum(terms)

Cnn = (partial)**2.*Pnlin+dic['Ngg']

betaf = 2.*deltac*(bg-1.)
B = betaf
C = 4*deltac*((deltac/a1**2.)*(b20-2.*(a1+a2)*(bg-1.))-2.*(bg-1.))
A = (2./a1**2.)*(deltac*(b20-2*(a1+a2)*(bg-1.))-a1**2.*(bg-1.))+2.*deltac*(bg-1.)

derpartialfnl = bg*((bg*dic['Ngg']/dic['Ngphiphi'])+A*(dic['Ngg']/dic['Ngb11'])+B*(dic['Ngg']/dic['Ngb01'])+2*fnlfid*C*(dic['Ngg']/dic['Ngb02']))
dfnlCnn = 2*partial*derpartialfnl*Pnlin

dfnlCgn = (bg*derpartialfnl)*Pnlin+(betaf/Mval)*partial*Pnlin

fbb = fore.getcompleteFisher(Cgg, Cgn, Cnn, derb_Pg, derb_Pgn, derb_Pn, derb_Pg, derb_Pgn, derb_Pn) #gg, gn, nn,
ffnlfnl = fore.getcompleteFisher(Cgg, Cgn, Cnn, derfnl_Pg, derfnl_Pgn, derfnl_Pn, derfnl_Pg, derfnl_Pgn, derfnl_Pn) #gg, gn, nn,
ffnlb = fore.getcompleteFisher(Cgg, Cgn, Cnn, derfnl_Pg, derfnl_Pgn, derfnl_Pn, derb_Pg, derb_Pgn, derb_Pn)
ffnlb2 = fore.getcompleteFisher(Cgg, Cgn, Cnn, derfnl_Pg, derfnl_Pgn, derfnl_Pn, derb2_Pg, derb2_Pgn, derb2_Pn)
fbb2 = fore.getcompleteFisher(Cgg, Cgn, Cnn, derb_Pg, derb_Pgn, derb_Pn, derb2_Pg, derb2_Pgn, derb2_Pn)
fb2b2 = fore.getcompleteFisher(Cgg, Cgn, Cnn, derb2_Pg, derb2_Pgn, derb2_Pn, derb2_Pg, derb2_Pgn, derb2_Pn)


fbb_gonly = fab(Cgg, derb_Pg, derb_Pg)
ffnlfnl_gonly = fab(Cgg, derfnl_Pg, derfnl_Pg)
ffnlb_gonly = fab(Cgg, derfnl_Pg, derb_Pg)

func_fbb_gonly = scipy.interpolate.interp1d(KK, fbb_gonly)
func_ffnlfnl_gonly = scipy.interpolate.interp1d(KK, ffnlfnl_gonly)
func_ffnlb_gonly = scipy.interpolate.interp1d(KK, ffnlb_gonly)

func_fbb = scipy.interpolate.interp1d(KK, fbb)
func_ffnlfnl = scipy.interpolate.interp1d(KK, ffnlfnl)
func_ffnlb = scipy.interpolate.interp1d(KK, ffnlb)
func_ffnlb2 = scipy.interpolate.interp1d(KK, ffnlb2) 
func_fbb2 = scipy.interpolate.interp1d(KK, fbb2)
func_fb2b2 = scipy.interpolate.interp1d(KK, fb2b2)

mat = np.zeros((3, 3))
mat_gonly = np.zeros((2, 2))

fint = []

errfnlmarg = []
errfnlmarg_gonly = []

kmin = np.min(KK)
kmax = np.max(KK)
V = (np.pi)**3./kmin**3.
Ks = np.arange(kmin, kmax/1.5, 0.001)

print('kmin, ', kmin, ' V,(Gpc^3) ', V/1e9)

limit = 100

for k in Ks:
    mat[0, 0] = scipy.integrate.quad(lambda x: func_fbb(x)*x**2., k, kmax, limit = limit)[0]
    mat[1, 1] = scipy.integrate.quad(lambda x: func_ffnlfnl(x)*x**2., k, kmax)[0]
    mat[0, 1] = scipy.integrate.quad(lambda x: func_ffnlb(x)*x**2., k, kmax)[0]
    mat[1, 0] = mat[0, 1]
    mat[2, 2] = scipy.integrate.quad(lambda x: func_fb2b2(x)*x**2., k, kmax)[0]
    mat[0, 2] = scipy.integrate.quad(lambda x: func_fbb2(x)*x**2., k, kmax)[0]
    mat[2, 0] = mat[0, 2]
    mat[1, 2] = scipy.integrate.quad(lambda x: func_ffnlb2(x)*x**2., k, kmax)[0]
    mat[2, 1] = mat[1, 2]
    mat *= V/(2*np.pi**2.)
    invmat = np.linalg.inv(mat)
    el = np.sqrt(invmat[1, 1])
    errfnlmarg += [el]
    mat_gonly[0, 0] = scipy.integrate.quad(lambda x: func_fbb_gonly(x)*x**2., k, kmax)[0]
    mat_gonly[1, 1] = scipy.integrate.quad(lambda x: func_ffnlfnl_gonly(x)*x**2., k, kmax)[0]
    mat_gonly[0, 1] = scipy.integrate.quad(lambda x: func_ffnlb_gonly(x)*x**2., k, kmax)[0]
    mat_gonly[1, 0] = mat_gonly[0, 1]    
    mat_gonly *= V/(2*np.pi**2.)
    invmat = np.linalg.inv(mat_gonly)
    el = np.sqrt(invmat[1, 1])
    errfnlmarg_gonly += [el]

func_f = func_ffnlfnl

fint = []

for k in Ks:
    mat = scipy.integrate.quad(lambda x: func_f(x)*x**2., k, kmax)[0]
    mat *= V/(2*np.pi**2.)
    fint += [mat]

fint = np.array(fint)
invf = fint**-0.5

func_f = func_ffnlfnl_gonly

fint = []

for k in Ks:
    mat = scipy.integrate.quad(lambda x: func_f(x)*x**2., k, kmax)[0]
    mat *= V/(2*np.pi**2.)
    fint += [mat]

fint = np.array(fint)
invf_gonly = fint**-0.5

plt.title('Forecasts, for $f_{nl}=$'+str(fnlfid)+', $bg=$'+str(bg)+', $b_{20}=$'+str(b20)+ ', $z=$'+str(z))
plt.xlabel('$K_{min}$ (Mpc$^{-1}$)')
plt.ylabel('$ Integrated \sigma(f_{nl}) $')
plt.loglog(Ks, invf_gonly, label = 'Non Marginalized Galaxy')
plt.loglog(Ks, errfnlmarg_gonly, label = 'Marginalized Galaxy, wrt $f_{nl},\ b$')
plt.loglog(Ks, errfnlmarg, label = 'Marginalized Combined, wrt $f_{nl},\ b,\ b2$')
plt.loglog(Ks, invf, label = 'Non Marginalized Combined')
plt.legend(loc = 'best', prop = {'size': 6})
plt.rc('grid', linestyle = "-", color = 'black')
plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-')
plt.minorticks_on()
plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2)
#plt.ylim(bottom = 5, top = 300)
plt.savefig(output+'fnlplotcomb.png', dpi = 300)
