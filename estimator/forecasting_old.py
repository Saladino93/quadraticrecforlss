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
        result = result[0]*V/(2.*np.pi**2.)
        return result

def getAllFisherElements(listdersPA, listdersPB, listdersPAB, PA, PB, PAB):
    Nvars = len(listdersPA)
    NKs = len(PA)
    AllFisherElements = np.zeros((Nvars, Nvars, NKs))
    for i in range(Nvars):
        for j in range(i, Nvars):
            der_i_PA = listdersPA[i]
            der_j_PA = listdersPA[j]
            der_i_PB = listdersPB[i]
            der_j_PB = listdersPB[j]
            der_i_PAB = listdersPAB[i]
            der_j_PAB = listdersPAB[j]
            fisherpermode_i_j = getcompleteFisher(PA, PAB, PB, der_i_PA, der_i_PAB, der_i_PB, der_j_PA, der_j_PAB, der_j_PB)
            AllFisherElements[i, j] = fisherpermode_i_j
    for k in range(NKs):
        M = AllFisherElements[:, :, k]
        AllFisherElements[:, :, k] = (M+M.T-np.diag(np.diag(M)))
    return AllFisherElements

def getMarginalizedCov(K, V, kmin, kmax, listdersPA, listdersPB, listdersPAB, PA, PB, PAB):
    Nvars = len(listdersPA)
    Ks = np.linspace(kmin, kmax/1.5, num = 20)
    NKs = len(Ks)
    AllFisherElements = getAllFisherElements(listdersPA, listdersPB, listdersPAB, PA, PB, PAB)
    mat = np.zeros((Nvars, Nvars, NKs))
    for i in range(Nvars):
        for j in range(i, Nvars):
            temp = []
            for k_minimum in Ks:
                temp += [getIntregratedFisher(K, AllFisherElements[i, j], k_minimum, kmax, V)]
            mat[i, j] = np.array(temp)
            
    MarginalizedCov = mat.copy()
    for k in range(NKs):
        M = mat[:, :, k]
        mat[:, :, k] = (M+M.T-np.diag(np.diag(M)))
        M = mat[:, :, k]
        MarginalizedCov[:, :, k] = np.linalg.inv(M)
    return MarginalizedCov
