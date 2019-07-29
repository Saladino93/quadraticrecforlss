import numpy as np

import scipy

import sympy as sp

keyfnl = 'fnl'


#matter domination
def D(z):
    return 1./(1.+z)


def getcompleteFisher(cgg, cgn, cnn, acgg, acgn, acnn):
    gg, gn, nn, agg, agn, ann = sp.symbols('gg gn nn agg agn ann') #agn = der_a(C^{gn})
    daC = sp.Matrix([[agg, agn], [agn, ann]])
    C = sp.Matrix([[gg, gn], [gn, nn]])
    invC = C.inv()
    prod = daC*invC*daC*invC
    tr = prod.trace()
    final = 0.5*sp.simplify(tr)
    expression = sp.lambdify([gg, gn, nn, agg, agn, ann], final, 'numpy')
    #expression = final.subs({gg: cgg, gn: cgn, nn: cnn, agg: acgg, agn: acgn, ann: acnn})
    result = expression(cgg, cgn, cnn, acgg, acgn, acnn)
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
