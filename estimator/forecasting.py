"""Computes various Fisher matrices using directly-measured and reconstructed fields in LSS.
"""

import sympy as sp

import numpy as np

import itertools

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "stix"

import scipy

import scipy.interpolate

import scipy.integrate

from scipy.signal import savgol_filter

import mpmath
mpmath.mp.prec = 100



#idea: to combine different forecasts just use + operation for object

class expression():
    # args here is just the list of variables used 
    def __init__(self, *args):
        var_list = []
        var_list_names = []
        ns = {}
        for a in args:
            symb = sp.symbols(a)
            setattr(self, a, symb)
            ns[a] = symb
            var_list += [symb]
            var_list_names += [a]
        self.vars = var_list
        self.vars_names = var_list_names
        self.ns = ns        

    def add_extra_var(self, name, expression):
        setattr(self, name, expression)

    # similar to add_extra_var but logic of usage is different  
    def add_expression(self, expression_name: str, expr):
        setattr(self, expression_name, sp.sympify(expr, locals = self.ns))
        
    def __add__(self, other_object_expression):
        return  

    def get_expression(self, expression_name):
        return getattr(self, expression_name)

    def evaluate_expression(self, expression_name, **valuesdict):
        expr = getattr(self, expression_name)
        expr_numpy = sp.lambdify(self.vars, expr, 'numpy')
        return expr_numpy(**valuesdict)

    def derivate(self, expression_name, variable_of_derivation, save_derivative = False):
        derivative = sp.diff(getattr(self, expression_name), variable_of_derivation) 
        if save_derivative:
            der_expression_name = 'der'+str(variable_of_derivation)+'_'+expression_name
            setattr(self, der_expression_name, derivative)
        return derivative

    def change_expression(self, expression_name, new_expression_value):
        setattr(self, expression_name, sp.sympify(new_expression_value, locals = self.ns))
        return


class Forecaster(expression):
    """Class that computes Fisher matrix and related quantities.
    """
    
    def __init__(self, K, *args):
        """
        Parameters
        ----------
        K : array
            List of k values to use in forecast.
        args : list
            List of variables we ultimately want to forecast for.
        """
        self.K = K
        self.length_K = len(K)
        expression.__init__(self, *args)

    def add_cov_matrix(self, covariance_matrix_dict, ):
        """Take covariance matrix from input file and store in convenient form.

        Parameters
        ----------
        covariance_matrix_dict : dictionary
            Dictionary defining auto and cross spectra that make up covariance
            matrix (e.g. {'Pgg' : ..., 'Pgn' : ..., 'Pnn' : ...}). Must have
            N(N+1)/2 keys for integer N.
        """
        elements = []
        expressions_list = []
        # Store expressions from covariance_dict
        for key, value in covariance_matrix_dict.items():
            self.add_expression(key, value)
            expressions_list += [self.get_expression(key)]
            elements += [key]
        # Store names of expressions
        self.covmatrix_str = elements

        # Assuming dict has N(N+1)/2 elements, extract N and make NxN matrix
        p = len(self.covmatrix_str)
        matrix_dim = int((-1+np.sqrt(1+8*p))/2)
        self.matrix_dim = matrix_dim

        shape = [matrix_dim, matrix_dim]
        covariance = sp.zeros(*shape)

        # Fill in covariance matrix elements
        for i in range(matrix_dim):
            covariance[i, :] = [expressions_list[i:matrix_dim+i]] 
        self.cov_matrix = covariance

 
    def plot_cov(self, var_values, legend = {'Plin': 'black'}, title = 'Covs', 
                 xlabel = '$K$ $(h Mpc^{-1})$', ylabel = '$P$ $(h^3 Mpc^{-3})$', output_name = ''):
        """Plot various ingredients of covariance matrix.

        Parameters
        ----------
        var_values : dict
            Dictionary of variable values to use in plots.
        legend : dict
            Dictionary of items to plot, specifying colors for each.
            Possible keys: Plin, Pgg, Pgn, Pnn, Ngg, shot
        title : str
            Title of plot.
        xlabel, ylabel : str
            Axis labels for plot.
        output_name : str
            File name for plot output.
        """
        K = self.K

        # Evaluate covariance matrix on specified values of parameters.
        all_vars = self.vars
        numpy_covariance_matrix = sp.lambdify(all_vars, self.cov_matrix, 'numpy') 
        temp_cov = numpy_covariance_matrix(**var_values)

        ##quicky, but it should be done like above
  
        spectra = {} 

        
        gg = temp_cov[0, 0, :]
        spectra['Pgg'] = gg
        try:
            nn = temp_cov[1, 1, :]
            gn = temp_cov[0, 1, :]
            spectra['Pgn'] = gn
            spectra['Pnn'] = nn
        except:
            a = 1

        if 'Plin' in legend.keys():
            Plin = var_values['Plin']
            spectra['Plin'] = Plin            
        
        spectra['shot'] = K*0.+1./var_values['nhalo']
        spectra['Ngg'] = var_values['Ngg']
          
        # Make plot
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale('log')
        for v in legend.keys():
            plt.plot(K, spectra[v], label = v, color = legend[v]['color'], ls = legend[v]['ls'], lw = 2)

        ax.legend(loc = 'best', prop = {'size': 6})
        fig.savefig(output_name, dpi = 300)
        plt.close(fig)        


    def __fab__(self, cgg, dera_cgg, derb_cgg):
        A = dera_cgg/cgg
        B = derb_cgg/cgg
        tot = 1./2.
        tot *= A*B
        return tot

    def __get__inv(self):
        return 0


    def __getF__(self, covariance_matrix, dera_covariance_matrix, derb_covariance_matrix, var_values = None):
 
        """Compute Fisher matrix element.

        Parameters
        ----------
        covariance_matrix : array
            Symbolic version of covariance matrix.
        dera_covariance_matrix : array
            Symbolic version of derivative covariance matrix w.r.t. parameter a.
        derb_covariance_matrix : array
            Symbolic version of derivative covariance matrix w.r.t. parameter b.
        numpify : bool, optional
            Whether to return numpy-computed Fisher matrix (default: False).
        var_values : dict, optional
            Dictionary of variable values for numerical computation.

        Returns
        -------
        final : array
            Fisher element
        """
        all_vars = self.vars


        #Possible improvements: everything in sympy. Just done in 2020branch
        #Problem: numerical precision --> have to use mpmath --> still have loops

        #this is an auxiliarly variable
        #sometimes you can have some matrix sympy entries that are just scalar and to not vectorize
        po = sp.symbols('po') 
        if po not in all_vars:
            all_vars += [po]
            var_values['po'] = np.ones(self.length_K)

        numpy_covariance_matrix = sp.lambdify(all_vars, covariance_matrix, 'numpy')
        numpy_dera_covariance_matrix = sp.lambdify(all_vars, dera_covariance_matrix+po*sp.ones(*covariance_matrix.shape), 'numpy')
        numpy_derb_covariance_matrix = sp.lambdify(all_vars, derb_covariance_matrix+po*sp.ones(*covariance_matrix.shape), 'numpy')
                          
        # Evaluate covariance and covariance derivatives at input parameter values
        cov_mat = numpy_covariance_matrix(**var_values)
        dera_cov_mat = numpy_dera_covariance_matrix(**var_values)
        derb_cov_mat = numpy_derb_covariance_matrix(**var_values)

        shape = cov_mat.shape

        final = []

        # Compute Fisher matrix
        for i in range(shape[-1]):
            cov = cov_mat[:, :, i]
            dera_cov = dera_cov_mat[:, :, i]-var_values['po'][i]*1
            derb_cov = derb_cov_mat[:, :, i]-var_values['po'][i]*1
            invC = np.linalg.inv(cov)
            prod = dera_cov@invC@derb_cov@invC
            final += [0.5*np.matrix.trace(prod)]

        final = np.array(final)

        return final 


    def get_fisher_matrix(self, variables_list = [], var_values = None, verbose = True):
        """Get per-k Fisher matrix, as a matrix of numpy functions.
          Symbolic form deprecated for now.
        Parameters
        ----------
        variables_list : list, optional
            List of variable names to include in Fisher matrix. If not specified,
            everything in self.vars is used.
        verbose : bool, optional
            Whether to print some status updates (default: True).
        """
        matrix_dim = self.matrix_dim
        shape = [matrix_dim, matrix_dim]

         # Compute derivatives of covariance matrix w.r.t. each parameter of interest
        for variable in self.vars:
            der_cov = sp.diff(self.cov_matrix, variable) 
            setattr(self, 'der'+str(variable)+'_matrix', der_cov)
    
        # Make list of parameters to use
        if variables_list == []:
            lista = self.vars
            s = ' all '         
        else:
            lista = variables_list
            s = ' '

        if verbose:
            print('Using'+s+'parameters list:', lista)

        self.fisher_list = lista

        combs = list(itertools.combinations_with_replacement(list(lista), 2))    
       
        N = len(lista)
        shape = [N, N]
        fab = sp.zeros(*shape)

        fab_numpy = np.zeros((N, N, self.length_K))

        if verbose:
            print('Calculating fisher matrix per mode')

        for a, b in combs:
            if verbose:
                print('\t%s, %s' % (a,b))
            dera_cov = self.get_expression('der'+str(a)+'_matrix')
            derb_cov = self.get_expression('der'+str(b)+'_matrix')
            i, j = lista.index(a), lista.index(b)
            f = self.__getF__(self.cov_matrix, dera_cov, derb_cov, var_values)
            fab_numpy[i, j, :] = f
            fab_numpy[j, i, :] = f                

     
        self.fisher = fab_numpy
        self.fisher_numpy = fab_numpy

        if verbose:
            print('Done calculating fisher matrix per mode')
 
            

    def get_error(self, variable, marginalized = False, integrated = False, kmin = 0.005, kmax = 0.05, volume = 100, Ks = None, verbose = True):
        """Get Fisher errorbar on specific parameter.

        Parameters
        ----------
        variable : str
            Parameter to get error for.
        marginalized : bool, optional
            Whether to marginalize over other parameters (default: False).
        integrated : bool, optional
            Whether to integrate Fisher matrix in k (default: False).
        kmin : float, optional
            Minimum value of long-mode k_min to use (default: 0.005).
        kmax : float, optional
            Maximum value of long-mode k_min to use (default: 0.05).
        volume : float, optional
            Survey volume, in (Gpc/h)^3 (default: 100).
        Ks: array, optional
            Kmin on which to calculate integrated value. (default: None) If None gets a value specified by object.

        NOTE: option marginalized = True and integrated = False not implemented for ill conditioned matrices due to numerical errors
            coming from machine precision. 
        """               

        if marginalized and not integrated:
            print('Marginalized per mode error not implemented!')
            return 0.

        # Convert volume from Gpc^3 to Mpc^3
        volume *= 10**9

        error = self.get_non_marginalized_error_per_mode(variable)            

        K = self.K.copy()

        if integrated:

             # Make array of k_min values to consider
            if Ks is None:
                Ks = np.arange(kmin, kmax, 0.001)

            # Define empty array for Fisher values, and fetch parameter pairs
            shape = self.fisher_numpy.shape
            f_int = np.zeros((shape[0], shape[1], len(Ks)))
            cov_int_marg = f_int.copy()
            lista = self.fisher_list
            combs = list(itertools.combinations_with_replacement(list(lista), 2))
 
            if verbose:
                print('Getting integrated error')

            # For each parameter pair, get Fisher element for each k, and
            # do k integral

            for a, b in combs:
                if verbose:
                    print('\t%s, %s' % (a,b))
                i, j = lista.index(a), lista.index(b)
                f = self.fisher_numpy[i, j, :]

                IntegratedFish = np.array([])
                np.savetxt('fish'+str(a+b)+'.txt', np.c_[K, f])
                for Kmin in Ks:
                    error = self.getIntegratedFisher(K, f, Kmin, kmax, volume)
                    IntegratedFish = np.append(IntegratedFish, error)
            
                f_int[i, j, :] = IntegratedFish
                f_int[j, i, :] = f_int[i, j, :]

            if verbose:
                print('Done getting integrated error')

            N = len(lista)
            np.save('matrix.npy', f_int)
            # For each k_min, invert the Fisher matrix
            for i in range(f_int.shape[-1]):
                matrix = f_int[..., i]
                cov_int_marg[..., i] = np.linalg.inv(matrix)
                error_inversion1 = np.max(abs(cov_int_marg[..., i]@matrix-np.eye(N)))
                error_inversion2 = np.max(abs(matrix@cov_int_marg[..., i]-np.eye(N)))
                # or can just check if m*minv = minv*m within some error
                if (error_inversion1 > 1e-2) or (error_inversion2 > 1e-2):
                    print('WARNING: Fisher matrix inversion not accurate.')

            

            ind1, ind2 = lista.index(variable), lista.index(variable)
            
            # Get the errorbar, as either F_{ii}^{-0.5} in the unmarginalized
            # case, or (F^{-1})_{ii}^{0.5} in the marginalized case
            if not marginalized:
                error = f_int[ind1, ind2, :]**-0.5
            else:
                error = cov_int_marg[ind1, ind2, :]**0.5 
 

            K = Ks
            error = error

        return K, error
   
    def get_non_marginalized_error_per_mode(self, variable):
        """Get unmarginalized error for a specific parameter.

        Parameters
        ----------
        variable : str
            Parameter to get error for.

        Returns
        -------
        error : float
            Unmarginalized errorbar on specified parameter.
        """
        lista = self.fisher_list
        i, j = lista.index(variable), lista.index(variable) 
        error = (self.fisher_numpy[i, j])**-0.5
        return error       


    def set_mpmath_integration_precision(self, integration_prec = 53):
        mpmath.mp.prec = integration_prec

    def getIntegratedFisher(self, K, FisherPerMode, kmin, kmax, V, apply_filter = False, scipy_mode = False):
        """Integrate per-mode Fisher matrix element in k, to get full Fisher matrix element.

        Given arrays of k values and corresponding Fisher matrix elements F(k), compute
            \frac{V}{(2\pi)^2} \int_{kmin}^{kmax} dk k^2 F(k) ,
        where V is the survey volume. The function actually first interpolates over
        the discrete supplied values of FisherPerMode, and then integrates the
        interpolating function between kmin and kmax using scipy.integrate.quad.

        Parameters
        ----------
        K : ndarray
            Array of k values.
        FisherPerMode : array
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
            print('\tMin, max K available: %f %f' % (np.min(K),np.max(K)))
            print('\tInput Kmin, Kmax: %f %f' % (kmin,kmax))
            return 0
        else:

            if apply_filter:
                lenK = len(K)
                window_length = 7 #2*int(lenK/10)+1 #make sure it is an odd number
                FisherPerMode = savgol_filter(FisherPerMode, window_length, 3) #, mode = 'nearest')

            function = scipy.interpolate.interp1d(K, FisherPerMode)

            if scipy_mode:
                result = scipy.integrate.quad(lambda x: function(x)*x**2., kmin, kmax, epsrel = 1e-15)
            else:
                ## A bit slow but it is worth it for numerical precision
                def f(x):
                    x = float(x)
                    y = function(x)*x**2.
                    return mpmath.mpf(1)*y

                resultmp = mpmath.quad(f, [kmin, kmax])
                result = [resultmp]

            result = result[0]*V/(4.*np.pi**2.)
            return result

    def plot_forecast(self, variable, error_versions, kmin = 0.005, kmax = 0.05,
                     volume = 100, title = 'Error', xlabel = '$K$ $(h Mpc^{-1})$', 
                    ylabel = '$\sigma$', xscale = 'linear', yscale = 'log', output_name = ''):
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.grid(which = 'both')

        for label, value in error_versions.items():
            K, error = self.get_error(variable, value['marginalized'], value['integrated'], 
                                      kmin, kmax, volume) 
            plt.plot(K, error, label = label, lw = 2)
        ax.legend(loc = 'best', prop = {'size': 6})
        fig.savefig(output_name, dpi = 300)
        plt.close(fig)
