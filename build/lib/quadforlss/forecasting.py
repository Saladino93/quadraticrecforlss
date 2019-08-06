import numpy as np

import scipy

import sympy as sp

keyfnl = 'fnl'


#matter domination
def D(z):
    return 1./(1.+z)


'''
b bias parameter on large scales
fnl fnl parameter on large scales
func function multiplying fnl
nbar number density of tracers
Pnl non linear power spectrum on large scales (approx Plinear on large scales)
'''
def definepowermodel(b, fnl, func, nbar, Pnl):
    return 0


def getcompleteFisher(cgg, cgn, cnn, acgg, acgn, acnn, bcgg = None, bcgn = None, bcnn = None):
    if bcgg is None:
        bcgg = acgg
        bcgn = acgn
        bcnn = acnn
    gg, gn, nn, agg, agn, ann, bgg, bgn, bnn = sp.symbols('gg gn nn agg agn ann bgg bgn bnn') #agn = der_a(C^{gn})
    daC = sp.Matrix([[agg, agn], [agn, ann]])
    dbC = sp.Matrix([[bgg, bgn], [bgn, bnn]])
    C = sp.Matrix([[gg, gn], [gn, nn]])
    invC = C.inv()
    prod = daC*invC*dbC*invC
    tr = prod.trace()
    final = 0.5*sp.simplify(tr)
    expression = sp.lambdify([gg, gn, nn, agg, agn, ann, bgg, bgn, bnn], final, 'numpy')
    result = expression(cgg, cgn, cnn, acgg, acgn, acnn, bcgg, bcgn, bcnn)
    return result


def getFisherpermode(el1, el2, k, mu, Pgg, Pnn, Pgn, PLfid, fnlfid = 0, cfid = 1, bgfid = 1, bnfid = 1, kappafid = 0):
    if (el1 == keyfnl) and (el2 == keyfnl):
        r = Pgn/np.sqrt(Pgg*Pnn)

        derPggdfnl = 2.*(bgfid+cfid*fnlfid/k**2.)*(cfid/k**2.)*PLfid
        derPgndfnl = cfid*bnfid*(1./k**2.)*PLfid

        term1 = (derPggdfnl/Pgg)-2.*r**2.*(derPgndfnl/Pgn)
        term1 = term1**2.
        term2 = 2.*r**2.*(1.-r**2.)*(derPgndfnl/Pgn)**2.

        term = term1+term2
        term /= (2.*(1.-r**2.)**2.)

        return term

def getFisherpermodeggonly(el1, el2, k, mu, Pgg, PLfid, fnlfid = 0, cfid = 1, bgfid = 1):
        if (el1 == keyfnl) and (el2 == keyfnl):

            derPggdfnl = 2.*(bgfid+cfid*fnlfid/k**2.)*(cfid/k**2.)*PLfid
               
            term = (derPggdfnl/Pgg)**2.
            term /= 2.

            return term


#check code
#this is for fnl = 0 and Pgg signal >> Pgg noise
def getFisherpermodefnlfid0(el1, el2, k, mu, Pgg, Pnn, Pgn, cfid = 1, bgfid = 1, bnfid = 1, kappafid = 0):
        if (el1 == keyfnl) and (el2 == keyfnl):
            r = Pgn/np.sqrt(Pgg*Pnn)
            term = (2.-r**2.)/(1.-r**2.)
            term *= (cfid/(k**2.*bgfid))**2.                
            return term


def getIntregratedFisher(K, FisherPerMode, kmin, kmax, V):
    if (kmin<np.min(K)) or (kmax>np.max(K)):
        print('Kmin(Kmax) should be higher(lower) than the minimum(maximum) of the K avaliable!')
        return 0
    else:
        function = scipy.interpolate.interp1d(K, FisherPerMode)
        result = scipy.integrate.quad(lambda x: function(x)*x**2., kmin, kmax)		
        result = result[0]*V/(2.*np.pi)**2.
        return result
