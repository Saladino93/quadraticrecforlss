import numpy as np
import sympy as sp
from scipy.interpolate import interp1d, interp2d
import scipy.interpolate as si
from scipy import integrate
import itertools
import functools

from mpmath import mp
import vegas

import warnings
warnings.filterwarnings('error')

class vectorize(np.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

def vectorize_2dinterp(function, x1, x2, min_x1, max_x1, fill_value):

    x1 = np.array(x1)
    x2 = np.array(x2)
    
    A = x1<max_x1
    B = min_x1<x1
    #C = x2<max_x2
    #D = min_x2<x2
    
    #E = A*B#*C*D
    mask = A*B
    o = mask.astype(float)
    o[~mask] = fill_value
    mask = o
    #mask = np.ones(x1.shape)
    #mask[~E] = fill_value
    
    temp = si.dfitpack.bispeu(function.tck[0], function.tck[1], function.tck[2], function.tck[3], function.tck[4], x1, x2)[0]
    
    result = mask*temp
    
    return result

def vectorize_2dinterp_bivariate(ev, x1, x2, min_x1, max_x1, fill_value):

    x1 = np.array(x1)
    x2 = np.array(x2)
    
    A = x1<max_x1
    B = min_x1<x1

    E = A*B
    mask = E

    o = mask.astype(float)
    o[~mask] = fill_value
    mask = o
    
    temp = ev(x2, x1) #NOTE IS SWAPPED HERE!
    
    result = mask*temp    
    
    return result


class Estimator(object):

#Power is an array with the total power spectrum of the observed field
#calculated in several k bins

    def __init__(self, minkhrec, maxkhrec, tot_k = 0., tot_mu = 0., tot_power = 0., Plin = 0., Pnlinsign = 0., bias = 0., nhalo = 0.):

        self.min_k_rec = minkhrec
        self.max_k_rec = maxkhrec

        Pnlinsign_scipy2d = interp2d(tot_k, tot_mu, Pnlinsign, fill_value = 0., bounds_error = False)
        self.Pnlinsign_scipy2d = Pnlinsign_scipy2d

        Plinsign_times_bias = (bias)*Plin

        Plinsign_times_bias_oned = (bias[0, :])*Plin #ASSUMES NO MU DEPENDENCE, useful for tris shot noise bispec part, faster
        tck = si.splrep(tot_k, Plinsign_times_bias_oned)

        def Pxbias(k):
            return si.splev(k, tck)

        self.Plinsign_times_bias_scipy_oned = Pxbias

        ev = si.RectBivariateSpline(tot_mu, tot_k, Plinsign_times_bias).ev

        def func(x1, x2):

            x1 = np.array(x1)
            x2 = np.array(x2)
            
            A = x1<max_x1
            B = min_x1<x1
            #C = x2<max_x2
            #D = min_x2<x2
            
            E = A*B#*C*D
            
            #mask = np.ones(x1.shape)
            #mask[~E] = 0.
            mask = E
            
            temp = ev(x2, x1)
            
            result = mask*temp
            
            return result

        self.Plinsign_times_bias_scipy = func

        fill_value = 0.

        #bias2d = interp2d(tot_k, tot_mu, bias, fill_value = 0., bounds_error = False) 
        #self.bias = give_function(bias2d, fill_value)


        index_max = np.where(tot_k<maxkhrec)[0][-1]
        index_min = np.where(tot_k>minkhrec)[0][0]

        tot_power = tot_power[:, index_min:index_max] #slice so that scipy interp takes care of filling np.inf
        Plin = Plin[index_min:index_max]
        Pnlinsign = Pnlinsign[:, index_min:index_max]
        tot_k = tot_k[index_min:index_max]
        bias = bias[:, index_min:index_max]



        self.thP = interp1d(tot_k, Plin, fill_value = 0., bounds_error = False)#This is for the Numerator 

        min_x1, max_x1, min_x2, max_x2 = tot_k.min(), tot_k.max(), tot_mu.min(), tot_mu.max()

        def give_function(P2d, fill_value):
            def give_P(q, mu):
                return vectorize_2dinterp(P2d, q, mu, min_x1, max_x1, fill_value)
            return give_P

        def give_function_bivariate(P2dev, fill_value):
            def give_P(q, mu):
                return vectorize_2dinterp_bivariate(P2dev, q, mu, min_x1, max_x1, fill_value)
            return give_P

        
        Ptot2_interp = interp2d(tot_k, tot_mu, tot_power, kind = 'cubic', fill_value = np.inf) #Goes into the Denominator of the Filter
        fill_value = np.inf
        #self.P = lambda q, mu: vectorize_2dinterp(Ptot2_interp, q, mu, min_x1, max_x1, min_x2, max_x2, fill_value)
        self.P = give_function(Ptot2_interp, fill_value)

        Pnlinsign_scipy2d = interp2d(tot_k, tot_mu, Pnlinsign, fill_value = 0., bounds_error = False)
        Pidentity_scipy2d = interp2d(tot_k, tot_mu, Pnlinsign*0.+1., fill_value = 0., bounds_error = False) 
        #Plinsign_times_bias_scipy2d = interp2d(tot_k, tot_mu, Plinsign_times_bias, fill_value = 0., bounds_error = False)
        
        fill_value = 0.
        
        self.Pnlinsign_scipy = give_function(Pnlinsign_scipy2d, fill_value)

        self.Pidentity_scipy = give_function(Pidentity_scipy2d, fill_value)

        #self.Plinsign_times_bias_scipy_2 = give_function(Plinsign_times_bias_scipy2d, fill_value)

        self.nhalo = nhalo

        self._F = {}
        self.c_a = {}
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


    def addF(self, key, F, ca = None):
        self._F[key] = F
        self._Ffunc[key] = sp.lambdify([self.q1, self.q2, self.mu], self._F[key], 'numpy')
        if ca is not None:
            self.c_a[key] = ca
        self.keys = self._F.keys()

     
    def getF(self, key):
        return self._F[key]

    def getF_func(self, key):
        return self._Ffunc[key]

    def addvalues(self, keys, values):
        if not (len(keys) == len(values)):
            print('Length of keys does not much that of values')
            pass  
        for k, v in zip(keys, values):
            self._values[k] = v

    def getvalues(self, key):
        return self._values[key]

    #q_1, q_2 moduli, mu12 angle between
    #2*[F(q1+q2, -q1)P(q1)+1<->2]
    def f_general(self, a, triple_1, triple_2):

        qtot, thetatot, phitot, mutot = self._sum_vectors(triple_1, triple_2)
        triple_tot = (qtot, thetatot, phitot)

        q1, q2 = triple_1[0], triple_2[0]

        minus_triple_1 = (triple_1[0], np.pi+triple_1[1], np.pi-triple_1[2])
        minus_triple_2 = (triple_2[0], np.pi+triple_2[1], np.pi-triple_2[2])
 
        mu1 = self._get_mu_between(triple_tot, minus_triple_1) 
        mu2 = self._get_mu_between(triple_tot, minus_triple_2)

        Fkernel = self._Ffunc[a]#sp.lambdify([self.q1, self.q2, self.mu], self._F[a], 'numpy')

        result = Fkernel(qtot, q1, mu1)*self.thP(q1)+Fkernel(qtot, q2, mu2)*self.thP(q2)
        result *= 2

        return result


    def f(self, a, q, K, mu):
        #if mu_sign = -1, mu should be already corrected for it outside
        #use f general, it is better and more bullet proof
        #Note if K --> -K, expressions should be the same, as it is accounted in mu sign change and kernel is ok for this

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

            mu_p = K**2.-q*K*mu
            mu_p /= (K*modK_q)

            result *= f(q, modK_q, mu)
            result /= (2*self.P(q, mu)*self.P(modK_q, mu_p))

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

           mu_p = K**2.-q*K*mu
           mu_p /= (K*modK_q)

           result *= f(a, q, K, mu)*f(b, q, K, mu)
           result /= (2*self.P(q, mu)*self.P(modK_q, mu_p))

           return result
        return _integrand


    def _outer_integral_vegas(self, f, K, a, b):
        @vegas.batchintegrand
        def _integrand(x):
           mu = x[:, 0]
           q = x[:, 1]

           modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)

           result = 2*np.pi*q**2./(2*np.pi)**3.

           mu_p = K**2.-q*K*mu
           mu_p /= (K*modK_q)

           result *= f(a, q, K, mu)*f(b, q, K, mu)


           #Pqmu = vectorize_2dinterp(self.P, q, mu)
           #PmodK_qmu_p = vectorize_2dinterp(self.P, modK_q, mu_p)
           #div = np.array(Pqmu*PmodK_qmu_p)
 
           result /= (2*self.P(q, mu)*self.P(modK_q, mu_p))
           return result
        return _integrand

    

    def _outer_integral_vegas_for_g(self, function1, function2, f, K, mu_sign, a, extra_g = False): 
        #for int g_a * function, where g_a is the weigh #for int g_a * function, where g_a is the weightt
        '''
        Integral of the type: \frac{1}{(2\pi)^3}\int d^3\vec{q} g_{\alpha}(\vec{q}, \vec{K}-\vec{q})f(\vec{q},\vec{K})

        If mu_sign = -1 it considers K --> -K 

        function1 is calculated at q
        function2 is calculated at \vec{K}-\vec{q}

        If extra_g = True, then there is a g_{\alpha}^2 rather than a g
        '''

        @vegas.batchintegrand
        def _integrand(x):
           mu = x[:, 0]*mu_sign
           q = x[:, 1]

           modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu) #if mu_sign negative here it is like a sum

           result = 2*np.pi*q**2./(2*np.pi)**3. #2pi is the integration over angle on plane perpendicular to K already done

           mu_p = K**2.-q*K*mu #if mu_sign negative here it is like a sum
           mu_p /= (K*modK_q)*mu_sign #that that mu_p is sign of -K-q relative to K

           result *= f(a, q, K, mu)*function1(q, mu)*function2(modK_q, mu_p)
           result /= (2*self.P(q, mu)*self.P(modK_q, mu_p))

           if extra_g:
               result *= f(a, q, K, mu)
               result /= (2*self.P(q, mu)*self.P(modK_q, mu_p))

           return result
        return _integrand


    def _double_outer_integral_vegas_for_g(self, function, f, K, subtract_qtot = True, mu_sign = 1, mu_sign_prime = 1, a = 'g'): 
        '''
        Integral of the type int_q int_q' g_a(q, K-q)g_a(q', K-q')f(K-(q+q'))

        mu_sign = -1 flips angle between K and q, or q -->-q or K --> -K

        mu_sign_prime = -1, same for q'

        If subtract_qtot = False, then you have f(q+q')
   
        '''


        @vegas.batchintegrand
        def _integrand(x):
           phi = x[:, 0] #angle between q and K
           phi_prime = x[:, 1] #angle between q' and K
           mu = np.cos(x[:, 0])*mu_sign
           mu_prime = np.cos(x[:, 1])*mu_sign_prime
           theta = x[:, 2]
           theta_prime = x[:, 3]
           q = x[:, 4]
           q_prime = x[:, 5]

           modK_q = np.sqrt(K**2.+q**2.-2*K*q*mu)
           
           modK_q_prime = np.sqrt(K**2.+q_prime**2.-2*K*q_prime*mu_prime)

           result = q**2./(2*np.pi)**3.*q_prime**2./(2*np.pi)**3.

           #qtot = (q*np.sin(theta)*np.cos(phi)+q_prime*np.sin(theta_prime)*np.cos(phi_prime))**2.
           #qtot += (q*np.sin(theta)*np.sin(phi)+q_prime*np.sin(theta_prime)*np.sin(phi_prime))**2.
           qtot = (q*np.sin(theta)*np.sin(phi)+q_prime*np.sin(theta_prime)*np.sin(phi_prime))**2.
           qtot += (q*np.cos(theta)*np.sin(phi)+q_prime*np.cos(theta_prime)*np.sin(phi_prime))**2.
           q_radial2 = (q*np.cos(phi)+q_prime*np.cos(phi_prime))**2.
           q_radial = np.sqrt(q_radial2) #parallel to direction of K
           qtot += q_radial2
           qtot = np.sqrt(qtot)

           mutot = q_radial/qtot

           mu_p = K**2.-q*K*mu
           mu_p /= (K*modK_q)*mu_sign #mu_sign, otherwise you have K+q and not -K-q

           mu_prime_p = K**2.-q_prime*K*mu_prime
           mu_prime_p /= (K*modK_q_prime)*mu_sign_prime

           if subtract_qtot:
               modK_qtot = np.sqrt(K**2.+qtot**2.-2*K*qtot*mutot)
               qtot = modK_qtot
               mutot = K**2.-qtot*K*mutot
               mutot /= (K*modK_qtot)

           result *= f(a, q, K, mu)*f(a, q_prime, K, mu_prime)*function(qtot, mutot)
           result /= (2*self.P(q, mu)*self.P(modK_q, mu_p)*2*self.P(q_prime, mu_prime)*self.P(modK_q_prime, mu_prime_p))

           return result
        return _integrand


    def _double_outer_integral_vegas_for_g2(self, function, K, subtract_qtot = True, mu_sign = 1, mu_sign_prime = 1, a = 'g'): 
        '''
        Integral of the type int_q int_q' g_a(q, K-q)g_a(q', K-q')f(K-(q+q'))

        If subtract_qtot = False, then you have f(q+q')
   
        '''


        @vegas.batchintegrand
        def _integrand(x):
           
           phi = x[:, 0] #angle between q and K
           phi_prime = x[:, 1] #angle between q' and K
           mu = np.cos(x[:, 0])
           mu_prime = np.cos(x[:, 1])
           theta = x[:, 2]
           theta_prime = x[:, 3]
           q = x[:, 4]
           q_prime = x[:, 5]
      
           

           zero_vec = np.ones(len(q))*0.
           triple_K = (zero_vec+K, zero_vec+0., zero_vec+0.)
           triple_minus_K = (zero_vec+K, zero_vec+0., zero_vec+np.pi)
           triple_q_prime = (q_prime, theta_prime, phi_prime)
           triple_minus_q_prime = (q_prime, np.pi+theta_prime, np.pi-phi_prime)
           triple_q = (q, theta, phi)
           triple_minus_q = (q, np.pi+theta, np.pi-phi)
           qtot, theta_tot, phi_tot, mutot = self._sum_vectors(triple_minus_K, triple_minus_q_prime)
           triple_minus_K_minus_q_prime = (qtot, theta_tot, phi_tot)

           qtot, theta_tot, phi_tot, mutot = self._sum_vectors(triple_q, triple_q_prime)
           # -(q+q')
           triple_minus_q_plus_q_prime = (qtot, np.pi+theta_tot, np.pi-phi_tot)

           qtot, theta_tot, phi_tot, mutot = self._sum_vectors(triple_K, triple_minus_q)
           triple_K_minus_q = (qtot, theta_tot, phi_tot)

           qtot, theta_tot, phi_tot, mutot = self._sum_vectors(triple_q, triple_q_prime)
           # (q+q')
           triple_q_plus_q_prime = (qtot, theta_tot, phi_tot)
           

           #K-(q+q')
           qtot, theta_tot, phi_tot, mutot = self._sum_vectors(triple_minus_q_plus_q_prime, triple_K)
           triple = (qtot, theta_tot, phi_tot)

           result = q**2./(2*np.pi)**3.*q_prime**2./(2*np.pi)**3.
           result *= self.f_general(a, triple_q, triple_K_minus_q)*self.f_general(a, triple_q_prime, triple_minus_K_minus_q_prime)

           if subtract_qtot:
               qtot, theta_tot, phi_tot = triple #K-(q+q')
           else:
               qtot, theta_tot, phi_tot = triple_q_plus_q_prime #(q+q')
           mutot = np.cos(phi_tot)
           
           result *= function(qtot, mutot)

           modK_q = triple_K_minus_q[0]
           mu_p = np.cos(triple_K_minus_q[2])

           mod_minus_K_minus_q_prime = triple_minus_K_minus_q_prime[0]
           mu_prime_p = np.cos(triple_minus_K_minus_q_prime[2])

           result /= (2*self.P(q, mu)*self.P(modK_q, mu_p)*2*self.P(q_prime, mu_prime)*self.P(mod_minus_K_minus_q_prime, mu_prime_p))

           return result
        return _integrand



    @vectorize
    def integrate_for_shot(self, a, K, mu_sign, minq, maxq, function1, function2, vegas_mode = False, extra_g = False):

        if vegas_mode:

            nitn = 100
            neval = 1000

            function = self._outer_integral_vegas_for_g(function1, function2, self.f, K, mu_sign, a, extra_g)

            integ = vegas.Integrator([[-1, 1], [minq, maxq]], nhcube_batch = 1000)
            result = integ(function, nitn = nitn, neval = neval)
            integral = result.mean

        return integral


    @vectorize
    def double_integrate_for_shot(self, a, K, mu_sign, minq, maxq, mu_sign_prime, minq_prime, maxq_prime, function, subtract_qtot = True, vegas_mode = False, version2 = False):

        if vegas_mode:

            nitn = 100
            neval = 1000

            if version2:#newer version
                integrand = self._double_outer_integral_vegas_for_g2(function = function, K = K, subtract_qtot = subtract_qtot, a = a)
            else:
                integrand = self._double_outer_integral_vegas_for_g(function, self.f, K, subtract_qtot, mu_sign, mu_sign_prime, a)
            integ = vegas.Integrator([[0, np.pi], [0, np.pi], [0, 2*np.pi], [0, 2*np.pi], [minq, maxq], [minq, maxq]], nhcube_batch = 1000)
            result = integ(integrand, nitn = nitn, neval = neval)
            integral = result.mean

        else:
            raise NotImplemented

        return integral


    @vectorize
    def double_integrate_bispectrum_for_shot(self, a, K, minq, maxq, minq_prime, maxq_prime, vegas_mode = False):

        if vegas_mode:

            nitn = 100
            neval = 1000

            function = self._double_outer_integrand_bispectrum_for_shot(K, a)
            integ = vegas.Integrator([[0, np.pi], [0, np.pi], [0, 2*np.pi], [0, 2*np.pi], [minq, maxq], [minq, maxq]], nhcube_batch = 2000)
            result = integ(function, nitn = nitn, neval = neval)
            integral = result.mean

        else:
            raise NotImplemented

        return integral

    def _double_outer_integrand_bispectrum_for_shot(self, K, key = 'g'):
        '''
        Integral of the type int_q int_q' g_a(q, K-q)g_a(q', K-q')sum_ B(k1, k2, k3)
        '''

        @vegas.batchintegrand
        def _integrand(x):
           phi = x[:, 0] #angle between q and K
           phi_prime = x[:, 1] #angle between q' and K
           mu = np.cos(x[:, 0])
           mu_prime = np.cos(x[:, 1])
           theta = x[:, 2]
           theta_prime = x[:, 3]
           q = x[:, 4]
           q_prime = x[:, 5]
           
           #mu_K = 1. #parallel to K
           #mu_minus_K = -1.

           zero_vec = np.ones(len(q))*0.
            
           triple_K = (zero_vec+K, zero_vec+0., zero_vec+0.)
           triple_minus_K = (zero_vec+K, zero_vec+0., zero_vec+np.pi) 
           triple_q_prime = (q_prime, theta_prime, phi_prime)
           triple_minus_q_prime = (q_prime, np.pi+theta_prime, np.pi-phi_prime)

           triple_q = (q, theta, phi)
           triple_minus_q = (q, np.pi+theta, np.pi-phi)
           

           qtot, theta_tot, phi_tot, mutot = self._sum_vectors(triple_minus_K, triple_minus_q_prime)
           triple_minus_K_minus_q_prime = (qtot, theta_tot, phi_tot)

           qtot, theta_tot, phi_tot, mutot = self._sum_vectors(triple_K, triple_minus_q)
           triple_K_minus_q = (qtot, theta_tot, phi_tot)

           qtot, theta_tot, phi_tot, mutot = self._sum_vectors(triple_q, triple_q_prime)
           triple_q_plus_q_prime = (qtot, theta_tot, phi_tot)

           # -(q+q')
           triple_minus_q_plus_q_prime = (qtot, np.pi+theta_tot, np.pi-phi_tot)


           #B(k1, k2, k3, mu1, mu2, mu3, mu_k1k2, mu_k1k3, mu_k2k3)
           #calc B(K, q', -K-q')

           #mu between q' and -K-q'
           mu_minus_K_minus_q_prime_q_prime = self._get_mu_between(triple_minus_K_minus_q_prime, triple_q_prime)
           #assert(self._get_mu_between(triple_q_prime, triple_minus_K_minus_q_prime) == mu_minus_K_minus_q_prime_q_prime)

           #mu between K and -K-q'
           mu_minus_K_minus_q_prime_K = self._get_mu_between(triple_minus_K_minus_q_prime, triple_K)

           #mu between q' and K
           mu_q_prime_K = self._get_mu_between(triple_q_prime, triple_K)

           a = triple_K
           b = triple_q_prime
           c = triple_minus_K_minus_q_prime

           muab = mu_q_prime_K
           muac = mu_minus_K_minus_q_prime_K
           mubc = mu_minus_K_minus_q_prime_q_prime
           
           #calc B(K, q', -K-q')
           Bg_1 = self._get_tot_bispectrum_signal(a[0], b[0], c[0], np.cos(a[2]), np.cos(b[2]), np.cos(c[2]), muab, muac, mubc)

           
           #calc B(-(q+q'), q, q')

           a = triple_minus_q_plus_q_prime
           b = triple_q
           c = triple_q_prime

           muab = self._get_mu_between(a, b)
           muac = self._get_mu_between(a, c)
           mubc = self._get_mu_between(b, c)

           #calc B(-(q+q'), q, q')
           Bg_2 = self._get_tot_bispectrum_signal(a[0], b[0], c[0], np.cos(a[2]), np.cos(b[2]), np.cos(c[2]), muab, muac, mubc)
        
           #calc B(-K, q, K-q)

           a = triple_minus_K
           b = triple_q
           c = triple_K_minus_q

           muab = self._get_mu_between(a, b)
           muac = self._get_mu_between(a, c)
           mubc = self._get_mu_between(b, c)

           #calc B(-K, q, K-q)
           Bg_3 = self._get_tot_bispectrum_signal(a[0], b[0], c[0], np.cos(a[2]), np.cos(b[2]), np.cos(c[2]), muab, muac, mubc)


           B_g = Bg_1+4*Bg_2+Bg_3 #Bg_2 is for 4 times s several terms are identical by substituion and symmetry

           result = q**2./(2*np.pi)**3.*q_prime**2./(2*np.pi)**3.
           
           result *= self.f_general(key, triple_q, triple_K_minus_q)*self.f_general(key, triple_q_prime, triple_minus_K_minus_q_prime)*B_g

           modK_q = triple_K_minus_q[0]
           mu_p = np.cos(triple_K_minus_q[2])

           mod_minus_K_minus_q_prime = triple_minus_K_minus_q_prime[0]
           mu_prime_p = np.cos(triple_minus_K_minus_q_prime[2])

           result /= (2*self.P(q, mu)*self.P(modK_q, mu_p)*2*self.P(q_prime, mu_prime)*self.P(mod_minus_K_minus_q_prime, mu_prime_p))

           return result
        return _integrand





    def get_bispectrum_shot_noise(self, a, K = None, minq = None, maxq = None, vegas_mode = True, verbose = True):

        if minq is None:
            minq = self.min_k_rec
        if maxq is None:
            maxq = self.max_k_rec
        if K is None:
            K = self.Krange
            murange = self.murange 
        else:
            murange = np.linspace(-1, 1, len(K))

        if verbose:
            print(f'Calculating shot noise of cross correlation with input field for {a} estimator.')

        self.bispectrum_shot = self._get_bis_shot_noise(a, K, murange, minq, maxq, vegas_mode = vegas_mode)
            
        return self.bispectrum_shot

    def _get_bis_shot_noise(self, a, K, mus, minq, maxq, vegas_mode = True):
        '''
        function1 is calculated at q
        function2 is calculated at \vec{K}-\vec{q}
        '''

        Naa = self.getN(a, a, K)

        shot = 1/self.nhalo

        #int g_a(q, K-q) * 1/n**2
        shotfactor_zeroPpower = self.integrate_for_shot(a, K, mu_sign = 1, minq = minq, maxq = maxq, function1 = self.Pidentity_scipy, function2 = self.Pidentity_scipy, vegas_mode = vegas_mode, extra_g = False)
        sh_bis_1 = (Naa*shotfactor_zeroPpower)*shot**2.

        #int g_a(q, K-q) * 1/n * P(K)
        sh_bis_3 = (shotfactor_zeroPpower*Naa)*self.Pnlinsign_scipy2d(K, mus)*shot

        shotfactor_onePpower = self.integrate_for_shot(a, K, mu_sign = 1, minq = minq, maxq = maxq, function1 = self.Pnlinsign_scipy, function2 = self.Pidentity_scipy, vegas_mode = vegas_mode, extra_g = False)

        #This one here gives:
        #int g_a(q, K-q)P(q) and
        #int g_a(q, K-q)P(K-q) = int g_a(K-q', q')P(q') with q'=K-q and volume of int is the same
        #Both are equal thanks to symmetry of filter
        sh_bis_2 = (shotfactor_onePpower*Naa)*shot

        sh_bis = sh_bis_1+2*sh_bis_2+sh_bis_3 

        return sh_bis

    #@vectorize
    def _bispectrum_second_order_part(self, k1, k2, mu1, mu2, mu_k1k2):
        '''
        Gives 2*b1**2.*[sum_a c_a * F_a(k1, k2)*P_lin(k1)*P_lin(k2)]
        '''
        
        second_order = 0.
        #for k in self.keys:
        #    second_order += self.c_a[k]*self.getF_func(k)(k1, k2, mu_k1k2) 

        second_order = self.Ftot_func(k1, k2, mu_k1k2) 

        #second_order *= 2*self.Plinsign_times_bias_scipy(k1, mu1)*self.Plinsign_times_bias_scipy(k2, mu2)
        
        second_order *= 2*self.Plinsign_times_bias_scipy_oned(k1)*self.Plinsign_times_bias_scipy_oned(k2) #for faster calc

        #second_order *= 2*self.bias(k1, mu1)*self.bias(k2, mu2)
        #second_order *= self.thP(k1)*self.thP(k2) #self.thP is Plin

        return second_order

    @vectorize
    def _get_tot_bispectrum_signal(self, k1, k2, k3, mu1, mu2, mu3, mu_k1k2, mu_k1k3, mu_k2k3):
        '''
        Calculates B_g(k1, k2, k3) = 2*b1**2.*[sum_a c_a * F_a(k1, k2)*P_lin(k1)*P_lin(k2)]+2 perms([1,3], [2,3])
        '''
        
        Bg1 = self._bispectrum_second_order_part(k1, k2, mu1, mu2, mu_k1k2)
        Bg2 = self._bispectrum_second_order_part(k1, k3, mu1, mu3, mu_k1k3)
        Bg3 = self._bispectrum_second_order_part(k2, k3, mu2, mu3, mu_k2k3)
        Bg = Bg1+Bg2+Bg3

        return Bg


    @vectorize
    def _get_dot_product(self, x1, x2):

        a1, a2, a3 = x1
        b1, b2, b3 = x2

        return a1*b1+a2*b2+a3*b3


    def _to_cartesian(self, q, theta, phi):

        x = q*np.cos(theta)*np.sin(phi)
        y = q*np.sin(theta)*np.sin(phi)
        z = q*np.cos(phi)

        return (x, y, z)

    @vectorize
    def _get_dot_product_sferical(self, triple1, triple2):

        q, theta, phi = triple1
        q_prime, theta_prime, phi_prime = triple2

        x1 = self._to_cartesian(q, theta, phi)
        x2 = self._to_cartesian(q_prime, theta_prime, phi_prime)

        return self._get_dot_product(x1, x2)

    #@vectorize
    def _sum_vectors(self, triple1, triple2):

        x1x, x1y, x1z = self._unpack_triple_to_cartesian(triple1)
        x2x, x2y, x2z = self._unpack_triple_to_cartesian(triple2)

        #qtot = (q*np.sin(theta)*np.sin(phi)+q_prime*np.sin(theta_prime)*np.sin(phi_prime))**2.
        #qtot += (q*np.cos(theta)*np.sin(phi)+q_prime*np.cos(theta_prime)*np.sin(phi_prime))**2.
        #q_radial2 = (q*np.cos(phi)+q_prime*np.cos(phi_prime))**2.

        qtotx2 = (x1x+x2x)**2.
        qtoty2 = (x1y+x2y)**2.
        q_radial2 = (x1z+x2z)**2. #qtotz2

        qtot2 = qtotx2+qtoty2+q_radial2

        qtot = np.sqrt(qtot2)
        q_radial = np.sqrt(q_radial2) #parallel to direction of K
        qtotx = np.sqrt(qtotx2)
        qtoty = np.sqrt(qtoty2)

        mutot = q_radial/qtot

        phitot = np.arccos(mutot)

        theta_tot = np.arctan2(qtoty, qtotx) #note it is defined on -pi, pi
        theta_tot = theta_tot*(theta_tot>0)+(2*np.pi+theta_tot)*(theta_tot<0) #convert to interval for spherical coordinates

        return qtot, theta_tot, phitot, mutot #modulus, cosine of angle wrt K, angle on the plane perpendicular to K=radial or z direction (I do not call radial direction of summed vector here)


    def _unpack_triple_to_cartesian(self, triple, to_array = False):
        q, theta, phi = triple
        x_v = self._to_cartesian(q, theta, phi)
        x, y, z = x_v
        if to_array:
            result = np.array([x, y, z])
        else:
            result = (x, y, z)
        return result

    def _get_mu_between(self, triple1, triple2):
        '''
        Give angle between q1 and q2.
        '''
        x1 = self._unpack_triple_to_cartesian(triple1, to_array = True)
        x2 = self._unpack_triple_to_cartesian(triple2, to_array = True)

        diff_module2 = np.sum((x1-x2)**2) #|q|^2 = (diff1**2.+...), diffi = x_i-y_i, i component
        
        x1mod2 = np.sum(x1**2.)
        x2mod2 = np.sum(x2**2.)

        x1mod = np.sqrt(x1mod2)
        x2mod = np.sqrt(x2mod2)

        mu_between = (x1mod2+x2mod2-diff_module2)/(2*x1mod*x2mod)

        return mu_between


    
    def get_trispectrum_shot_noise(self, a, K = None, minq = None, maxq = None, vegas_mode = True, verbose = True):
        if verbose:
            print(f'Calculating shot noise of autocorrelation of reconstruced field for {a} estimator.')

        Ftot = 0.
        for k in self.keys:
            Ftot += self.c_a[k]*self.getF(k)

        self.Ftot = Ftot 
        self.Ftot_func = sp.lambdify([self.q1, self.q2, self.mu], Ftot, 'numpy')

        if minq is None:
            minq = self.min_k_rec
        if maxq is None:
            maxq = self.max_k_rec
        if K is None:
            K = self.Krange
            murange = self.murange
        else:
            murange = np.linspace(-1, 1, len(K))

        self.trispectrum_shot = self._get_tris_shot_noise(a, K, murange, minq, maxq, vegas_mode = vegas_mode)

        return self.trispectrum_shot



    def _get_tris_shot_noise(self, a, K, mus, minq, maxq, vegas_mode = True):

        Naa = self.getN(a, a, K)

        shot = 1/self.nhalo

        #function1 is function of just q
        #function2 is function of K-q

        # int g_a(q, K-q)P(q)
        # mode is q+K-q=K
        A11 = self.integrate_for_shot(a, K, mu_sign = 1, minq = minq, maxq = maxq, function1 = self.Pnlinsign_scipy, function2 = self.Pidentity_scipy, vegas_mode = vegas_mode)

        # int g_a(q, -K-q)
        # mode is q-K-q=-K
        A12 = self.integrate_for_shot(a, K, mu_sign = -1, minq = minq, maxq = maxq, function1 = self.Pidentity_scipy, function2 = self.Pidentity_scipy, vegas_mode = vegas_mode)

        A1 = A11*A12*Naa**2. #Naa**p, a power of one for each g_a

        # int g_a(q, -K-q)P(q)
        # mode is q-K-q=-K
        A21 = self.integrate_for_shot(a, K, mu_sign = -1, minq = minq, maxq = maxq, function1 = self.Pnlinsign_scipy, function2 = self.Pidentity_scipy, vegas_mode = vegas_mode)

        # int g_a(q, K-q)
        # mode is q+K-q=K
        A22 = self.integrate_for_shot(a, K, mu_sign = 1, minq = minq, maxq = maxq, function1 = self.Pidentity_scipy, function2 = self.Pidentity_scipy, vegas_mode = vegas_mode)

        A2 = A21*A22*Naa**2. #Naa**p, a power of one for each g_a
  
        # int g_a(q, K-q)P(K-q)
        A31 = self.integrate_for_shot(a, K, mu_sign = 1, minq = minq, maxq = maxq, function1 = self.Pidentity_scipy, function2 = self.Pnlinsign_scipy, vegas_mode = vegas_mode)

        # int g_a(q, -K-q)
        # mode is q-K-q=-K
        A32 = A12

        A3 = A31*A32*Naa**2. #Naa**p, a power of one for each g_a

        # int g_a(q, -K-q)P(-K-q)
        # mode is q-K-q=-K
        A41 = self.integrate_for_shot(a, K, mu_sign = -1, minq = minq, maxq = maxq, function1 = self.Pidentity_scipy, function2 = self.Pnlinsign_scipy, vegas_mode = vegas_mode)

        # int g_a(q, K-q)
        A42 = A22

        A4 = A41*A42*Naa**2. #Naa**p, a power of one for each g_a

        A = (A1+A2+A3+A4)

        ############

        # P(K) * int g_a(q, K-q) * int g_a(q, -K-q) 
        B11 = self.Pnlinsign_scipy2d(K, mus)
        B12 = A22 #int g_a(q, K-q)
        B13 = A12 #int g_a(q, -K-q) 
        B1 = B11*B12*B13*Naa**2.


        # int g_a(q, K-q) g_a(q', -K-q') P(q-K-q')
        # 
        # note for P isotropic P(q-K+q')=P(K-(q+q'))
        B31 = self.double_integrate_for_shot(a, K, mu_sign = 1, minq = minq, maxq = maxq, mu_sign_prime = -1, minq_prime = minq, maxq_prime = maxq, function = self.Pnlinsign_scipy, subtract_qtot = True, vegas_mode = vegas_mode)
        B3 = B31*Naa**2.
        
        #B311 = self.double_integrate_for_shot(a, K, mu_sign = 1, minq = minq, maxq = maxq, mu_sign_prime = -1, minq_prime = minq, maxq_prime = maxq, function = self.Pnlinsign_scipy, subtract_qtot = True, vegas_mode = vegas_mode, version2 = True)
        #print(np.max(np.abs(B31-B311)/B311))
   
      
        # int g_a(q, K-q) g_a(q', -K-q') P(q-K+q')
        # note for P isotropic P(q-K+q')=P(K-(q+q'))
        B51 = self.double_integrate_for_shot(a, K, mu_sign = 1, minq = minq, maxq = maxq, mu_sign_prime = -1, minq_prime = minq, maxq_prime = maxq, function = self.Pnlinsign_scipy, subtract_qtot = False, vegas_mode = vegas_mode)
        B5 = B51*Naa**2.

        B = (B1+B3+B5) #NOTE B0=0 as it is delta of dirac at K

        n2_term = (A+B)*shot**2.

        # int g_a(q, -K-q)
        # mode is q-K-q=-K
        C = A12

        # int g_a(q, K-q)
        # MODE IS q+K-q=K
        D = A22
 
        n3_term = (C*D*Naa**2.)*shot**3
        
        #Then you also have Bispectrum terms

        F = self.double_integrate_bispectrum_for_shot(a, K, minq, maxq, minq, maxq, vegas_mode = vegas_mode)
        F *= Naa**2.
        
        n1_term = (F)*shot

        result = n1_term+n2_term+n3_term

        #print(n1_term)
        #print(n2_term)
        #print(n3_term)

        return result

    @vectorize
    def N(self, a, b, K, minq, maxq, vegas_mode = False):

        if vegas_mode:

            nitn = 100
            neval = 1000

            function = self._outer_integral_vegas(self.f, K, a, b)
            integ = vegas.Integrator([[-1, 1], [minq, maxq]], nhcube_batch = 1000)
            result = integ(function, nitn = nitn, neval = neval)
            integral = result.mean
        else:
            function = self._outer_integral(self.f, K, a, b)
            options = {'limit' : 100, 'epsrel': 1e-4}
            ris = integrate.nquad(function, [[minq, maxq], [-1., 1.]], opts = [options, options])
            ris = ris[0]
            integral = ris

        return integral**-1.


    def generateNs(self, K, mu, minq = None, maxq = None, listKeys = None, vegas_mode = False, verbose = True):

        if minq is None:
            minq = self.min_k_rec
        if maxq is None:
            maxq = self.max_k_rec

        values = self.keys

        if verbose:
            print('Keys are, ', self.keys)

        if listKeys is None:
            listKeys = list(itertools.combinations_with_replacement(list(values), 2))

        if verbose:
            print('Key combs to calculate is, ', listKeys)
        
        retList = {}
        retList_interpolated = {}

        for key1, key2 in listKeys:
            retList[key1+","+key2] = []

        for a, b in listKeys:
            if verbose:
                print('Computing N integral for (%s,%s)' % (a,b))
            N = self.N(a, b, K, minq, maxq, vegas_mode)
            retList[a+","+b]= N#.append(N) #if I do not vectorize generateNs I could assign retList the whole N, without append
            Ninterp = si.interp1d(K, N)
            retList_interpolated[a+","+b] = Ninterp
            retList_interpolated[b+","+a] = retList_interpolated[a+","+b]

        #for a, b in listKeys:
        #    retList[b+","+a] = np.array(retList[a+","+b])

        self.Nmatrix = retList
        self.Nmatrix_interpolated = retList_interpolated
        self.Krange = K
        self.murange = mu

        return None

    @vectorize
    def getN(self, a, b, K = None, interpolated = True):
        if K is None:
            K = self.Krange
        try:
            if interpolated:
                result = self.Nmatrix_interpolated[a+","+b](K)
            else:
                Kindex = np.where(self.Krange == K)[0]
                result = self.Nmatrix[a+","+b][Kindex]
            return result
        #except KeyError:
        #    return self.Nmatrix[b+","+a][Kindex] #N is symm, so it is useless, but I leave for completeness
        except:
            print("Key combination not found")
            raise

    def saveNs(self, filename):
        lista = self.keys
        listKeys = list(itertools.combinations_with_replacement(list(lista), 2))

        N = self.Krange
        for a, b in listKeys:
            N = np.c_[N, self.getN(a, b, K = self.Krange)]
        np.savetxt(filename, N)

        return None

