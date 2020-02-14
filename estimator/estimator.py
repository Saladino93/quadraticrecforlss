import numpy as np
import sympy as sp
from scipy.interpolate import interp1d
from scipy import integrate
import itertools
import functools

from mpmath import mp
import vegas

class vectorize(np.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)


class Estimator(object):

#Power is an array with the total power spectrum of the observed field
#calculated in several k bins

    def __init__(self, tot_k = 0., tot_power = 0., Plin = 0.):

        self.thP = interp1d(tot_k, Plin)#This is for the Numerator 
        self.P = interp1d(tot_k, tot_power)#Goes into the Denominator of the Filter

        self._F = {}
        self._Ffunc = {}
        self.q1, self.q2, self.mu = sp.symbols('q1 q2 mu')
        self._values = {}

        self._expr = {}
        self._exprfunc = {}

        self.keys = []
        return None

    def memoize(func):
        cache = func.cache = {}
        @functools.wraps(func)
        def memoized_func(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]
        return memoized_func


    def addF(self, key, F):
        self._F[key] = F
        self._Ffunc[key] = sp.lambdify([self.q1, self.q2, self.mu], self._F[key], 'numpy')
        self.keys = self._F.keys()

     
    def getF(self, key):
        return self._F[key]

    def addvalues(self, keys, values):
        if not (len(keys) == len(values)):
            print('Length of keys does not much that of values')
            pass  
        for k, v in zip(keys, values):
            self._values[k] = v

    def getvalues(self, key):
        return self._values[key]

    #q_1, q_2 moduli, suppose vec(q_2)= q_2(0, 0, 1), mu angle theta between vectors
    def f(self, a, q, K, mu):

        modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)

        mu_p = K**2.-q*K*mu
        mu_p /= (K*modK_q)

        Fkernel = self._Ffunc[a]#sp.lambdify([self.q1, self.q2, self.mu], self._F[a], 'numpy')

        result = Fkernel(q, K, -mu)*self.thP(q)+Fkernel(modK_q, K, -mu_p)*self.thP(modK_q)
        result *= 2

        return result

    def addexpression(self, key, expr):
        self._expr[key] = expr
        self._exprfunc[key] = sp.lambdify([self.q1, self.q2, self.mu], self._expr[key], 'numpy')


    @vectorize
    def integrateexpression(self, key, K, minq, maxq):
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

        options = {'limit' : 50}

        ris = integrate.nquad(function, [[minq, maxq], [-1., 1.]], opts = [options, options])
        ris = ris[0]

        integral = ris

        return integral**-1.

    def _outer_integral(self, f, K, a, b):
        def _integrand(q, mu):

           modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)

           result = 2*np.pi*q**2./(2*np.pi)**3.

           result *= f(a, q, K, mu)*f(b, q, K, mu)
           result /= (2*self.P(q)*self.P(modK_q))

           return result
        return _integrand


    def _outer_integral_vegas(self, f, K, a, b):
        @vegas.batchintegrand
        def _integrand(x):
           mu = x[:, 0]
           q = x[:, 1]

           modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)

           result = 2*np.pi*q**2./(2*np.pi)**3.

           result *= f(a, q, K, mu)*f(b, q, K, mu)
           result /= (2*self.P(q)*self.P(modK_q))

           return result
        return _integrand


    def _outer_integral_vegas_for_g(self, function1, function2, f, K, mu_sign, a): #for int g_a * function, where g_a is the weigh #for int g_a * function, where g_a is the weightt
        @vegas.batchintegrand
        def _integrand(x):
           mu = x[:, 0]*mu_sign
           q = x[:, 1]

           modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)

           result = 2*np.pi*q**2./(2*np.pi)**3.

           result *= f(a, q, K, mu)*function1(q)*function2(modK_q)
           result /= (2*self.P(q)*self.P(modK_q))

           return result
        return _integrand


    def _double_outer_integral_vegas_for_g(self, function, f, K, mu_sign, mu_sign_prime, a): #for int g_a * function, where g_a is the weigh #for int g_a * function, where g_a is the weightt
        @vegas.batchintegrand
        def _integrand(x):
           phi = x[:, 0]
           phi_prime = x[:, 1]
           mu = np.cos(x[:, 0])*mu_sign
           mu_prime = np.cos(x[:, 1])*mu_sign
           theta = x[:, 2]
           theta_prime = x[:, 3]
           q = x[:, 4]
           q_prime = x[:, 5]

           modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)
           
           modK_q_prime = np.sqrt(K**2.+q_prime**2.-2*K*q_prime*mu_sign_prime)

           result = q**2./(2*np.pi)**3.*q_prime**2./(2*np.pi)**3.

           qtot = (q*np.sin(theta)*np.cos(phi)+q_prime*np.sin(theta_prime)*np.cos(phi_prime))**2.
           qtot += (q*np.sin(theta)*np.sin(phi)+q_prime*np.sin(theta_prime)*np.sin(phi_prime))**2.
           qtot += (q*np.cos(phi)+q_prime*np.cos(phi_prime))**2.
           qtot = np.sqrt(qtot)

           result *= f(a, q, K, mu)*f(a, q_prime, K, mu_prime)*function(qtot)
           result /= (2*self.P(q)*self.P(modK_q)*2*self.P(q_prime)*self.P(modK_q_prime))

           return result
        return _integrand


    @vectorize
    def integrate_for_shot(self, a, K, mu_sign, minq, maxq, function1, function2, vegas_mode = False):

        if vegas_mode:

            nitn = 100
            neval = 1000

            function = self._outer_integral_vegas_for_g(function1, function2, self.f, K, mu_sign, a)
            integ = vegas.Integrator([[-1, 1], [minq, maxq]], nhcube_batch = 100)
            result = integ(function, nitn = nitn, neval = neval)
            integral = result.mean

        return integral


    @vectorize
    def double_integrate_for_shot(self, a, K, mu_sign, minq, maxq, mu_sign_prime, minq_prime, maxq_prime, function, vegas_mode = False):

        if vegas_mode:

            nitn = 100
            neval = 1000

            function = self._double_outer_integral_vegas_for_g(function, self.f, K, mu_sign, mu_sign_prime, a)
            integ = vegas.Integrator([[0, np.pi], [0, np.pi], [0, 2*np.pi], [0, 2*np.pi], [minq, maxq], [minq, maxq]], nhcube_batch = 1000)
            result = integ(function, nitn = nitn, neval = neval)
            integral = result.mean

        return integral



    @vectorize
    def N(self, a, b, K, minq, maxq, vegas_mode = False):

        if vegas_mode:

            nitn = 100
            neval = 1000

            function = self._outer_integral_vegas(self.f, K, a, b)
            integ = vegas.Integrator([[-1, 1], [minq, maxq]], nhcube_batch = 100)
            result = integ(function, nitn = nitn, neval = neval)
            integral = result.mean
 
        else:
            function = self._outer_integral(self.f, K, a, b)
            options = {'limit' : 100, 'epsrel': 1e-4}
            ris = integrate.nquad(function, [[minq, maxq], [-1., 1.]], opts = [options, options])
            ris = ris[0]
            integral = ris

        return integral**-1.


    def generateNs(self, K, minq, maxq, listKeys = None, vegas_mode = False, verbose = True):

        values = self.keys

        if verbose:
            print('Keys are, ', self.keys)

        if listKeys is None:
            listKeys = list(itertools.combinations_with_replacement(list(values), 2))

        if verbose:
            print('Key combs to calculate is, ', listKeys)
        
        retList = {}

        for key1, key2 in listKeys:
            retList[key1+","+key2] = []

        for a, b in listKeys:
            if verbose:
                print('Computing N integral for (%s,%s)' % (a,b))
            N = self.N(a, b, K, minq, maxq, vegas_mode)
            retList[a+","+b]= N#.append(N) #if I do not vectorize generateNs I could assign retList the whole N, without append

        for a, b in listKeys:
            retList[b+","+a] = np.array(retList[a+","+b])

        self.Nmatrix = retList
        self.Krange = K

        return None

    @vectorize
    def getN(self, a, b, K = None):
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
        lista = ['g', 's', 't']
        listKeys = list(itertools.combinations_with_replacement(list(lista), 2))

        N = self.Krange
        for a, b in listKeys:
            N = np.c_[N, self.getN(a, b, K = self.Krange)]
        np.savetxt(filename, N)

        return None

