"""Stores and computes quantities relating to quadratic estimators in LSS.
"""

import numpy as np
import sympy as sp
from scipy.interpolate import interp1d
from scipy import integrate
import itertools
import functools
from mpmath import mp


class vectorize(np.vectorize):
    """Vectorizes a function.
    """
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)


class Estimator(object):
    """Quadratic estimator class.
    """

    def __init__(self, tot_k = 0., tot_power = 0., Plin = 0.):
        """
        Parameters
        ----------
        tot_k : ndarray
            Array of k values corresponding to input power spectra.
        tot_power : ndarray
            Array of signal+noise power spectrum values (for denominator
            of quadratic estimator filter).
        Plin : ndarray
            Array of signal power spectrum values (for numerator of
            quadratic estimator filter).
        """

        self.thP = interp1d(tot_k, Plin)
        self.P = interp1d(tot_k, tot_power)

        self._F = {}
        self._Ffunc = {}
        self.q1, self.q2, self.mu = sp.symbols('q1 q2 mu')
        self._values = {}

        self._expr = {}
        self._exprfunc = {}

        self.keys = []

        return None


    def memoize(func):
        """Memoize a function (not currently used anywhere).
        """
        cache = func.cache = {}
        @functools.wraps(func)
        def memoized_func(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]
        return memoized_func


    def addF(self, key, F):
        """Add mode-coupling 'F' function to quadratic estimator.

        Parameters
        ----------
        key : str
            Key for mode-coupling (e.g. 't').
        F : sympy expression
            Algebraic form of mode-coupling F function (e.g. (2./7.)*est.mu**2.).
        """
        self._F[key] = F
        self._Ffunc[key] = sp.lambdify([self.q1, self.q2, self.mu], self._F[key], 'numpy')
        self.keys = self._F.keys()


    def getF(self, key):
        """Get stored expression for a particular mode-coupling function.

        Parameters
        ----------
        key : str
            Key for desired expression (e.g. 't').

        Returns
        -------
        exp : sympy expression
            Algebraic form of F function.
        """
        return self._F[key]

    def addvalues(self, keys, values):
        """Store values corresponding to specified keys.

        Parameters
        ----------
        keys : array
            Keys of values to store.
        values : array
            Values to store.
        """
        if not (len(keys) == len(values)):
            print('Length of keys does not much that of values')
            pass
        for k, v in zip(keys, values):
            self._values[k] = v

    def getvalues(self, key):
        """Return stored value corresponding to specified key.

        Parameters
        ----------
        key : str
            Key of value to return.

        Returns
        -------
        val :
            Stored value (unspecified format).
        """
        return self._values[key]

    #q_1, q_2 moduli, suppose vec(q_2)= q_2(0, 0, 1), mu angle theta between vectors
    def f(self, a, q, K, mu):
        """Numerical value of mode-coupling 'f' function.

        Computes
            f(k_1,k_2) = 2 [ F_a(k_1+k_2,-k_1) P_lin(|k_1|) + k_1 <--> k_2 ] ,
        where k_1 and k_2 are vectors, with |k_1+k_2| = K and |k_1| = q .

        Parameters
        ----------
        a : str
            Key for F function of interest.
        q : float
            Vector norm as described above.
        K : float
            Vector norm as described above.
        mu : float
            Angle between k_1+k_2 and k_1, defined as
                mu = (k_1+k_2) \cdot k_1 / ( |k_1+k_2| |k_1| ) .

        Returns
        -------
        result: float
            Value of f.
        """

        # Norm of K-q vector
        modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)

        # mu for K dot K-q
        mu_p = K**2.-q*K*mu
        mu_p /= (K*modK_q)

        Fkernel = self._Ffunc[a]

        result = Fkernel(q, K, -mu)*self.thP(q)+Fkernel(modK_q, K, -mu_p)*self.thP(modK_q)
        result *= 2

        return result

    def addexpression(self, key, expr):
        """Store expression corresponding to specified key.

        Stores both symbolic expression (in self._expr[key])
        and corresponding function (in self._exprfunc[key]).

        Parameters
        ----------
        key : str
            Key of expressions to store.
        expr : sympy expression
            Expression to store.
        """
        self._expr[key] = expr
        self._exprfunc[key] = sp.lambdify([self.q1, self.q2, self.mu], self._expr[key], 'numpy')


    @vectorize
    def integrateexpression(self, key, K, minq, maxq):
        """Integrate stored expression in wavenumber and angle.

        Specifically, if the stored expression is f(K,q,u), perform
            (2\pi)^{-3} \int_qmin^qmax dq 2 \pi q^2
                \int_{-1}^1 du \frac{f(K,q,u)}{2 P_tot(q) P_tot(|K-q|)}
        for fixed K using scipy.integrate.nquad (with hard-coded integration options),
        and with |K-q| understood to be the norm related to the vectors defined
        by K, q, and mu.

        The stored expression will actually be a function of vectors
        k_1 and k_2, but it is evaluated using arguments K, q, and mu,
        where K = |k_1+k_2|, q = |k_1|, and mu is the cosine of the angle
        between k_1+k_2 and k_1.

        Parameters
        ----------
        key : str
            Key for stored expression to integrate.
        K : float
            K value to evalute expression on.
        minq : float
            Lower limit for q integral.
        maxq : float
            Upper limit for q integral.

        Returns
        -------
        integral : float
            Result of integral.
        err : float
            Estimated uncertainty on numerical integration.
        """
        f = self._exprfunc[key]

        def function(q, mu):

            modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)
            result = 2*np.pi*q**2./(2*np.pi)**3.

            result *= f(q, modK_q, mu)
            result /= (2*self.P(q)*self.P(modK_q))

            return result

        options = {'limit' : 1000, 'epsrel': 1e-6, 'epsabs': 0.}

        ris = integrate.nquad(function, [[minq, maxq], [-1., 1.]], opts = [options, options])

        err = ris[1]
        integral = ris[0]

        return integral, err

    def _integrate(self, function, minq, maxq):
        """Integrate function in wavenumber and angle.

        Specifically, if function is f(q,u), perform
            I = \int_qmin^qmax dq \int_{-1}^1 du f(q,u)
        for fixed K using scipy.integrate.nquad (with hard-coded integration options),
        and return 1/I.

        Parameters
        ----------
        func : function
            Function to integrate, with arguments (q,mu).
        minq : float
            Lower limit for q integral.
        maxq : float
            Upper limit for q integral.

        Returns
        -------
        integral : float
            Result of one over the integral.
        """
        options = {'limit' : 50}

        ris = integrate.nquad(function, [[minq, maxq], [-1., 1.]], opts = [options, options])
        ris = ris[0]

        integral = ris

        return integral**-1.

    def _outer_integral(self, f, K, a, b):
        """Construct the integrand for an N_{ab} integral.

        Specifically, given a function f_a(q,K,mu), return the function
            \frac{2\pi q^2}{(2\pi)^3} f_a(q,K,mu) f_b(q,K,mu)
            \frac{1}{2 P_tot(q) P_tot(K-q)}
        with |K-q| understood to be the norm related to the vectors defined
        by K, q, and mu.

        Parameters
        ----------
        f : function
            Function with arguments (a,q,K,mu).
        K : float
            K value to feed to f.
        a : str
            Key for f_a.
        b : str
            Key for f_b.

        Returns
        -------
        integrand : function
            Python function for the integrand.
        """
        def _integrand(q, mu):

           modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)

           result = 2*np.pi*q**2./(2*np.pi)**3.

           result *= f(a, q, K, mu)*f(b, q, K, mu)
           result /= (2*self.P(q)*self.P(modK_q))

           return result

        return _integrand


    @vectorize
    def N(self, a, b, K, minq, maxq):
        """Compute N_{ab} by performing 2d integral.

        The integral is done with scipy.integrate.nquad, with hard-coded options.

        Parameters
        ----------
        a : str
            Key for first mode-coupling function.
        b : str
            Key for second mode-coupling function.
        K : float
            K value.
        minq : float
            lower limit for q integral.
        maxq : float
            upper limit for q integral.

        Returns
        -------
        result : float
            Numerical result for N_{ab}.
        """

        function = self._outer_integral(self.f, K, a, b)

        options = {'limit' : 100, 'epsrel': 1e-4}

        ris = integrate.nquad(function, [[minq, maxq], [-1., 1.]], opts = [options, options])
        ris = ris[0]

        integral = ris

        return integral**-1.


    def generateNs(self, K, minq, maxq, verbose = True):
        """Generate N_{ab} values at a list of K values and for all possible (a,b) pairs.

        The N matrix is stored as self.Nmatrix, and the K list as self.Krange.

        Parameters
        ----------
        K : ndarray(float)
            Array of K values to use.
        minq : float
            lower limit for q integral.
        maxq : float
            upper limit for q integral.
        verbose : bool, optional
            Whether to print some info during computation.
        """
        values = self.keys

        if verbose:
            print('Keys are, ', self.keys)

        listKeys = list(itertools.combinations_with_replacement(list(values), 2))

        retList = {}

        for key1, key2 in listKeys:
            retList[key1+","+key2] = []

        for a, b in listKeys:
            N = self.N(a, b, K, minq, maxq)
            retList[a+","+b]= N

        for a, b in listKeys:
            retList[b+","+a] = np.array(retList[a+","+b])

        self.Nmatrix = retList
        self.Krange = K

        return None

    @vectorize
    def getN(self, a, b, K = None):
        """Retrieve N_{ab} that has already been computed.

        Parameters
        ----------
        a : str
            Key for first mode-coupling function.
        b : str
            Key for second mode-coupling function.
        K : float, optional
            K value to retrieve at. If not specified, function uses entire K range
            that has been computed.

        Returns
        -------
        N : float or ndarray(float)
            N_{ab} value(s).
        """
        if K is None:
            K = self.Krange
        Kindex = np.where(self.Krange == K)[0]#[0]
        try:
            return self.Nmatrix[a+","+b][Kindex]
        except KeyError:
            return self.Nmatrix[b+","+a][Kindex]
        except:
            print("Key combination not found")
            raise

    def saveNs(self, filename):
        """Save N matrix for g,s,t mode-couplings to file.

        Parameters
        ----------
        filename : str
            File name for output.
        """
        lista = ['g', 's', 't']
        listKeys = list(itertools.combinations_with_replacement(list(lista), 2))

        N = self.Krange
        for a, b in listKeys:
            N = np.c_[N, self.getN(a, b, K = self.Krange)]
        np.savetxt(filename, N)

        return None
