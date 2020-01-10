"""Computes various Fisher matrices using directly-measured and reconstructed fields in LSS.
"""

import numpy as np
import scipy
import sympy as sp

# Key corresponding to f_NL.
keyfnl = 'fnl'



def D(z):
    """Linear growth factor during matter domination.

    Parameters
    ----------
    z : float or ndarray
        Redshift(s).

    Returns
    -------
    D : float or ndarray
        Linear growth factor(s).
    """
    return 1./(1.+z)


def getcompleteFisher(cgg, cgn, cnn, acgg, acgn, acnn):
    """Numerically compute Fisher matrix element for a single parameter.

    The user must supply gg, gn, and nn covariance matrices, along with
    their derivatives with respect to the parameter of interest. The function
    then computes
        F_{aa} = \frac{1}{2} Tr[ (\d_a C) C^{-1} (\d_a C) C^{-1} ] ,
    where \d_a denotes the derivative with respect to parameter a and the trace
    and matrix multiplications are taken over elements of C, where C is
        C = C_{gg} C_{gn}
            C_{gn} C_{nn} .

    Parameters
    ----------
    cgg : float
        gg covariance.
    cgn : float
        gn covariance.
    cnn : float
        nn covariance.
    acgg : float
        Derivative of gg covariance w.r.t. parameter a.
    acgn : float
        Derivative of gn covariance w.r.t. parameter a.
    acnn : float
        Derivative of nn covariance w.r.t. parameter a.

    Returns
    -------
    result : float
        Fisher matrix element as described above.
    """
    gg, gn, nn, agg, agn, ann = sp.symbols('gg gn nn agg agn ann') #agn = der_a(C^{gn})

    # Symbolically define C and \d_a C matrices
    daC = sp.Matrix([[agg, agn], [agn, ann]])
    C = sp.Matrix([[gg, gn], [gn, nn]])

    # Compute C^{-1} and (\d_a C) C^{-1} (\d_a C) C^{-1}, then take the trace
    # and multiply by 0.5
    invC = C.inv()
    prod = daC*invC*daC*invC
    tr = prod.trace()
    final = 0.5*sp.simplify(tr)

    # Evaluate the numerical result using the function inputs
    expression = sp.lambdify([gg, gn, nn, agg, agn, ann], final, 'numpy')
    #expression = final.subs({gg: cgg, gn: cgn, nn: cnn, agg: acgg, agn: acgn, ann: acnn})
    result = expression(cgg, cgn, cnn, acgg, acgn, acnn)
    return result


def getFisherpermode(el1, el2, k, mu, Pgg, Pnn, Pgn, PLfid,
                     fnlfid = 0, cfid = 1, bgfid = 1, bnfid = 1, kappafid = 0):
    """Compute Fisher matrix per k mode.

    Use Eq. (17) from draft, where
        \frac{d P_{gg}}{d f_{NL}} = 2 ( b_g + \frac{c f_NL}{k^2} ) \frac{c}{k^2} P_{lin}
    and
        \frac{d P_{gn}}{d f_{NL}} = 2 \frac{b_n c}{k^2} P_{lin}
    and
        \frac{d P_{nn}}{d f_{NL}} = 0 .

    Parameters
    ----------
    el1 : str
        Key for first parameter (only works for 'fnl').
    el2 : str
        Key for second parameter (only works for 'fnl').
    k : float
        K to evaluate at.
    mu : float
        Mu to evaluate at (irrelevant).
    Pgg : float
        Pgg value.
    Pnn : float
        Pnn value.
    Pgn : float
        Pgn value.
    PLfid : float
        Fiducal linear power spectrum value.
    fnlfid : float, optional
        Fiducal f_NL value (default: 0).
    cfid : float, optional
        Fiducial c value (default: 1).
    bgfid : float, optional
        Fiducial b_g value (default: 1).
    bnfid : float, optional
        Fiducial b_n value (default: 1).
    kappafid : float, optional
        Fiducial kappa value (not currently used; default: 0).

    Returns
    -------
    term : float
        Fisher matrix element per k mode, at specified k.
    """
    if (el1 == keyfnl) and (el2 == keyfnl):
        # TODO: raise exception otherwise

        # gn cross-correlation coefficient
        r = Pgn/np.sqrt(Pgg*Pnn)

        # d P_gg / d f_NL
        derPggdfnl = 2.*(bgfid+cfid*fnlfid/k**2.)*(cfid/k**2.)*PLfid
        # d P_gn / d f_NL
        # TODO: isn't there a bgfid missing below?
        derPgndfnl = cfid*bnfid*(1./k**2.)*PLfid

        term1 = (derPggdfnl/Pgg)-2.*r**2.*(derPgndfnl/Pgn)
        term1 = term1**2.
        term2 = 2.*r**2.*(1.-r**2.)*(derPgndfnl/Pgn)**2.

        # TODO: Code up terms 3 and 4 from Eq. (17) in draft, but add option
        # to omit them if we're assuming d P_nn / d f_NL = 0
        term = term1+term2
        term /= (2.*(1.-r**2.)**2.)

        return term

def getFisherpermodeggonly(el1, el2, k, mu, Pgg, PLfid, fnlfid = 0, cfid = 1, bgfid = 1):
    """Compute Fisher matrix per k mode, only including the gg covariance.

    Use Eq. (17) from draft, where
        \frac{d P_{gg}}{d f_{NL}} = 2 ( b_g + \frac{c f_NL}{k^2} ) \frac{c}{k^2} P_{lin}
    and
        \frac{d P_{gn}}{d f_{NL}} = \frac{d P_{nn}}{d f_{NL}} = 0 .

    Parameters
    ----------
    el1 : str
        Key for first parameter (only works for 'fnl').
    el2 : str
        Key for second parameter (only works for 'fnl').
    k : float
        K to evaluate at.
    mu : float
        Mu to evaluate at (irrelevant).
    Pgg : float
        Pgg value.
    PLfid : float
        Fiducal linear power spectrum value.
    fnlfid : float, optional
        Fiducal f_NL value (default: 0).
    cfid : float, optional
        Fiducial c value (default: 1).
    bgfid : float, optional
        Fiducial b_g value (default: 1).

    Returns
    -------
    term : float
        Fisher matrix element per k mode, at specified k.
    """
    if (el1 == keyfnl) and (el2 == keyfnl):
        # TODO: raise exception otherwise

        # d P_gg / d f_NL
        derPggdfnl = 2.*(bgfid+cfid*fnlfid/k**2.)*(cfid/k**2.)*PLfid

        term = (derPggdfnl/Pgg)**2.
        term /= 2.

        return term


#check code
#this is for fnl = 0 and Pgg signal >> Pgg noise
def getFisherpermodefnlfid0(el1, el2, k, mu, Pgg, Pnn, Pgn,
                            cfid = 1, bgfid = 1, bnfid = 1, kappafid = 0):
    """Compute Fisher matrix per k mode, for f_NL=0 and signal-dominated P_gg.

    See draft for expression. TODO: Is this expression self-consistent?

    Parameters
    ----------
    el1 : str
        Key for first parameter (only works for 'fnl').
    el2 : str
        Key for second parameter (only works for 'fnl').
    k : float
        K to evaluate at.
    mu : float
        Mu to evaluate at (irrelevant).
    Pgg : float
        Pgg value.
    Pnn : float
        Pnn value.
    Pgn : float
        Pgn value.
    cfid : float, optional
        Fiducial c value (default: 1).
    bgfid : float, optional
        Fiducial b_g value (default: 1).
    bnfid : float, optional
        Fiducial b_n value (default: 1).
    kappafid : float, optional
        Fiducial kappa value (not currently used; default: 0).

    Returns
    -------
    term : float
        Fisher matrix element per k mode, at specified k.
    """
    if (el1 == keyfnl) and (el2 == keyfnl):
        # TODO: raise exception otherwise

        r = Pgn/np.sqrt(Pgg*Pnn)

        term = (2.-r**2.)/(1.-r**2.)
        term *= (cfid/(k**2.*bgfid))**2.
        return term


def getIntregratedFisher(K, FisherPerMode, kmin, kmax, V):
    """Integrate per-mode Fisher matrix element in k, to get full Fisher matrix element.

    Given arrays of k values and corresponding Fisher matrix elements F(k), compute
        \frac{V}{2(2\pi)^2} \int_{kmin}^{kmax} \int_{-1}^1 du dk k^2 F(k) ,
    where V is the survey volume. The function actually first interpolates over
    the discrete supplied values of FisherPerMode, and then integrates the
    interpolating function between kmin and kmax using scipy.integrate.quad.

    Note that since our per-mode Fisher matrix doesn't depend on mu (written
    as u in the equation above), the mu integral is trivial, yielding a factor of 2.

    Parameters
    ----------
    K : ndarray
        Array of k values.
    FisherPerMode : ndarray
        Array of Fisher element values corresponding to K.
    kmin : float
        Lower limit of k integral.
    kmax : float
        Upper limit of k integral.
    V : float
        Survey volume.

    Returns
    -------
    result : float
        Result of integral.
    """
    if (kmin<np.min(K)) or (kmax>np.max(K)):
        print('Kmin(Kmax) should be higher(lower) than the minimum(maximum) of the K avaliable!')
        return 0
    else:
        function = scipy.interpolate.interp1d(K, FisherPerMode)
        result = scipy.integrate.quad(lambda x: function(x)*x**2., kmin, kmax)
        result = result[0]*V/(2.*np.pi)**2.
        return result
