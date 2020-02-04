"""Computes various Fisher matrices using directly-measured and reconstructed fields in LSS.
"""

import sympy as sp
import numpy as np
import itertools

import matplotlib.pyplot as plt

import scipy

import scipy.interpolate

import scipy.integrate

#what happens if I create another object and same var names but I want different values? Does sympy create two different instances?

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

    def add_cov_matrix(self, covariance_matrix_dict):
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
        #print(expressions_list)
        #print(covariance)
        #symm = covariance+covariance.T
        #covariance = sp.Matrix(matrix_dim, matrix_dim, lambda i, j: symm[i, j]/2 if i==j else symm[i, j])
        #print(covariance)
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
        nn = temp_cov[1, 1, :]
        gn = temp_cov[0, 1, :]

        spectra['Pgg'] = gg
        spectra['Pgn'] = gn
        spectra['Pnn'] = nn

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
            plt.plot(K, spectra[v], label = v, color = legend[v]['color'], ls = legend[v]['ls'])


        # ax.legend(loc = 'best', prop = {'size': 6})
        ax.legend(loc = 'best')
        # fig.savefig(output_name, dpi = 300)
        fig.savefig(output_name)
        plt.close(fig)


    def __fab__(self, cgg, dera_cgg, derb_cgg):
        A = dera_cgg/cgg
        B = derb_cgg/cgg
        tot = 1./2.
        tot *= A*B
        return tot

    def __get__inv(self):
        return 0

    def __getF__(self, covariance_matrix, dera_covariance_matrix, derb_covariance_matrix,
                 numpify = False, var_values = None):
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
        if numpify:
            all_vars = self.vars

            # TODO: what is po for? priors?
            po = sp.symbols('po')
            if po not in all_vars:
                all_vars += [po]
                var_values['po'] = np.ones(self.length_K)

            numpy_covariance_matrix = sp.lambdify(all_vars, covariance_matrix, 'numpy')
            numpy_dera_covariance_matrix = sp.lambdify(all_vars, dera_covariance_matrix+po*sp.ones(*covariance_matrix.shape), 'numpy') #sometimes you could have 0. or constant derivative from sympy --> do not vectorize
            numpy_derb_covariance_matrix = sp.lambdify(all_vars, derb_covariance_matrix+po*sp.ones(*covariance_matrix.shape), 'numpy')

            #numpy_covariance_matrix = np.vectorize(numpy_covariance_matrix)
            #numpy_dera_covariance_matrix= np.vectorize(numpy_dera_covariance_matrix)
            #numpy_derb_covariance_matrix = np.vectorize(numpy_derb_covariance_matrix)

            '''
            @np.vectorize
            def get_covmat(b10 = 1, b01 = 1, b11 = 1, b20 = 1, bs2 = 1, fnl = 1, nhalo = 1, Pnlin = np.ones(3), M = np.ones(3), deltac = 1, a1 = 1, a2 = 1, new_bias = 1, Nphiphig =  np.ones(3), Ngt = np.ones(3), Ngs = np.ones(3), Ngg = np.ones(3), Nc02g = np.ones(3), Nc01g = np.ones(3), Nc11g = np.ones(3)):
                return numpy_covariance_matrix(b10, b01, b11, b20, bs2, fnl, nhalo, Pnlin, M, deltac, a1, a2, new_bias, Nphiphig, Ngt, Ngs, Ngg, Nc02g, Nc01g, Nc11g)
            '''

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
        else:
            #Sympy way, non numerical. Fastish for simple expressions. More complex expressions can take 'infinite' time.
            #To be improved
            ##quick fix till i figure how i can do inv with all these vars as sympy gets stuck
            ##maybe just define general matrix and then subs values
            ##or calculate F directly on numbers!
            if covariance_matrix.shape == (2, 2):
                detC = covariance_matrix[0, 0]*covariance_matrix[1, 1]-covariance_matrix[0, 1]*covariance_matrix[1, 0]
                temp = sp.zeros(*covariance_matrix.shape)
                temp[0, 0] = covariance_matrix[1, 1]
                temp[0, 1] = -covariance_matrix[0, 1]
                temp[1, 0] = -covariance_matrix[1, 0]
                temp[1, 1] = covariance_matrix[0, 0]
                invC = temp/detC
            else:
                invC = covariance_matrix.inv()

            prod = dera_covariance_matrix*invC*derb_covariance_matrix*invC
            tr = prod.trace()
            final = 0.5*sp.simplify(tr)

        return final

    def get_fisher_matrix(self, variables_list = [], numpify = False, var_values = None, verbose = True):
        """Get per-k Fisher matrix, both in symbolic form and as a matrix of numpy functions.

        Parameters
        ----------
        variables_list : list, optional
            List of variable names to include in Fisher matrix. If not specified,
            everything in self.vars is used.
        numpify : bool, optional
            Whether to generate numpy-function version of Fisher matrix (default: False).
        verbose : bool, optional
            Whether to print some status updates (default: True).
        """

        '''
        for vv in var_values.keys():
            try:
                ll = len(var_values[vv])
                if ll > 1:
                    self.length_K = ll
                break
            except:
                a = 1 #dummy operation
        '''

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
            print('Calculating fisher matrix')

        # For each parameter pair, get Fisher element
        for a, b in combs:
            if verbose:
                print('\t%s, %s' % (a,b))
            dera_cov = self.get_expression('der'+str(a)+'_matrix')
            derb_cov = self.get_expression('der'+str(b)+'_matrix')
            i, j = lista.index(a), lista.index(b)

            f = self.__getF__(self.cov_matrix, dera_cov, derb_cov, numpify, var_values)
            if numpify:
                fab_numpy[i, j, :] = f
                fab_numpy[j, i, :] = f
            else:
                fab[i, j] = f

        if numpify:
            self.fisher = fab_numpy
            self.fisher_numpy = fab_numpy
        else:
            self.fisher = fab


    def get_error(self, variable, marginalized = False, integrated = False,
                  kmin = 0.005, kmax = 0.05, volume = 100):
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
        """

        # Convert volume from Gpc^3 to Mpc^3
        volume *= 10**9

        # Get unmarginalized error (only used if integrate==False)
        error = self.get_non_marginalized_error_per_mode(variable)

        K = self.K.copy()

        if integrated:

            # Make array of k_min values to consider
            Ks = np.arange(kmin, kmax, 0.001)

            # Define empty array for Fisher values, and fetch parameter pairs
            shape = self.fisher_numpy.shape
            f_int = np.zeros((shape[0], shape[1], len(Ks)))
            cov_int_marg = f_int.copy()
            lista = self.fisher_list
            combs = list(itertools.combinations_with_replacement(list(lista), 2))

            # For each parameter pair, get Fisher element for each k, and
            # do k integral
            for a, b in combs:
                i, j = lista.index(a), lista.index(b)
                f = self.fisher_numpy[i, j, :]

                IntegratedFish = np.array([])

                for Kmin in Ks:
                    error = self.getIntregratedFisher(K, f, Kmin, kmax, volume)
                    IntegratedFish = np.append(IntegratedFish, error)

                f_int[i, j, :] = IntegratedFish
                f_int[j, i, :] = f_int[i, j, :]

            # For each k_min, invert the Fisher matrix
            for i in range(f_int.shape[-1]):
                matrix = f_int[..., i]
                cov_int_marg[..., i] = np.linalg.inv(matrix)


            ind1, ind2 = lista.index(variable), lista.index(variable)

            # Get the errorbar, as either F_{ii}^{-0.5} in the unmarginalized
            # case, or (F^{-1})_{ii}^{0.5} in the marginalized case
            if not marginalized:
                # error = f_int[ind1, ind2, :]
                error = f_int[ind1, ind2, :]**-0.5
            else:
                # error = cov_int_marg[ind1, ind2, :]
                error = cov_int_marg[ind1, ind2, :]**0.5

            K = Ks
            # error = error**-0.5

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

    ###NOTE!
    ###REWRITE ROUTINE IN CLEANER WAY, CHECK POS DEF OF INTEGRATED FISHER, AND PER MODE WHAT HAPPENS
    def get_marginalized_error_per_mode(self, variable):
        lista = self.fisher_list

        inverse_fisher_numpy = self.fisher_numpy.copy()

        def is_pos_def(x):
            return np.all(np.linalg.eigvals(x) > 0)

        for i in range(self.fisher_numpy.shape[-1]):
            matrix = self.fisher_numpy[..., i]
            print(matrix)
            print(is_pos_def(matrix))
            matrix_inv = np.linalg.inv(matrix)
            inverse_fisher_numpy[..., i] = matrix_inv

        i, j = lista.index(variable), lista.index(variable)
        error = (self.inverse_fisher_numpy[i, j])**-0.5
        return error


    def get_integrated_error(self, fisher_vec, kmin = 0.005, kmax = 0.05, volume = 100):
        result = self.getIntregratedFisher(self.K, fisher_vec, kmin, kmax, volume)
        return result

    def getIntregratedFisher(self, K, FisherPerMode, kmin, kmax, V):
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
            function = scipy.interpolate.interp1d(K, FisherPerMode)
            result = scipy.integrate.quad(lambda x: function(x)*x**2., kmin, kmax)
            # result = result[0]*V/(2.*np.pi**2.)
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
        plt.grid(which="both")

        for label, value in error_versions.items():
            K, error = self.get_error(variable, value['marginalized'], value['integrated'],
                                      kmin, kmax, volume)
            # label = ''
            # for l, v in value.items():
            #     if v:
            #         label += l+' '
            plt.plot(K, error, label = label)
        ax.legend(loc = 'best')
        # ax.legend(loc = 'best', prop = {'size': 6})
        # fig.savefig(output_name, dpi = 300)
        fig.savefig(output_name)
        plt.close(fig)
