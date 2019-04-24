import numpy as np
import sympy as sp
from scipy.interpolate import interp1d
from scipy import integrate
import itertools
import functools


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


    def _outer_integral(self, K, a, b):
        def _integrand(q, mu):

           modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)

           result = 2*np.pi*q**2./(2*np.pi)**3.

           result *= self.f(a, q, K, mu)*self.f(b, q, K, mu)
           result /= (2*self.P(q)*self.P(modK_q))

           return result
        return _integrand


    @vectorize
    def N(self, a, b, K, minq, maxq):

        function = self._outer_integral(K, a, b)

	options = {'limit' : 50}

	ris = integrate.nquad(function, [[minq, maxq], [-1., 1.]], opts = [options, options])
        ris = ris[0]
	
	integral = ris

        return integral**-1.


    def generateNs(self, K, minq, maxq, verbose = True):

        values = np.array(self.keys)

	if verbose:
		print('Keys are, ', self.keys)

        listKeys = list(itertools.combinations_with_replacement(list(values), 2))
        
        retList = {}

        for key1, key2 in listKeys:
            retList[key1+","+key2] = []

        for a, b in listKeys:
            N = self.N(a, b, K, minq, maxq)
	    retList[a+","+b]= N#.append(N) #if I do not vectorize generateNs I could assign retList the whole N, without append

        for a, b in listKeys:
            retList[b+","+a] = np.array(retList[a+","+b])

        self.Nmatrix = retList
        self.Krange = K

        return None

    @vectorize	
    def getN(self, a, b, K):
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

