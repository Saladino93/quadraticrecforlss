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
import scipy.interpolate as si

import scipy.interpolate as si

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

    def __init__(self, K, priors, *args):
        """
        Parameters
        ----------
        K : array
            List of k values to use in forecast.
        args : list
            List of variables we ultimately want to forecast for.
        priors : dict
            Dictionary of Gaussian priors for variables.
        """
        self.K = K
        self.length_K = len(K)
        self.fisher_integrated = None
        self.fisher_integrated_marginalized = None
        self.inv_priors = {}
        for key, value in priors.items():
            if value == '':
                self.inv_priors[key] = 0.
            else:
                self.inv_priors[key] = 1./value**2.
        expression.__init__(self, *args)

    def add_cov_matrix(self, covariance_matrix_dict,
                       wedge_covariance_matrix_dict = None):
        """Take covariance matrix from input file and store in convenient form.

        Parameters
        ----------
        covariance_matrix_dict : dictionary
            Dictionary defining auto and cross spectra that make up covariance
            matrix (e.g. {'Pgg' : ..., 'Pgn' : ..., 'Pnn' : ...}). Must have
            N(N+1)/2 keys for integer N.
        wedge_covariance_matrix_dict : dictionary, optional
            As above, but for the covariance matrix for modes within a foreground
            wedge (defined by the mu_limit argument of Forecaster.get_error()).
            Default: None
        """
        elements = []
        expressions_list = []

        # Store expressions from covariance_matrix_dict
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
        self.cov_matrix_wedge = None
        self.cov_matrix_dim = matrix_dim

        # Go through same steps for wedge covariance
        # (could code this up in a less redundant way, but this is easiest for now)
        if wedge_covariance_matrix_dict is not None:
            elements_wedge = []
            expressions_list_wedge = []

            # Store expressions from wedge_covariance_matrix_dict,
            # if they have not already been stored
            for key, value in wedge_covariance_matrix_dict.items():
                if key not in covariance_matrix_dict.keys():
                    self.add_expression(key, value)
                expressions_list_wedge += [self.get_expression(key)]
                elements_wedge += [key]
            # Store names of expressions
            self.covmatrix_str_wedge = elements_wedge

            # Assuming dict has N(N+1)/2 elements, extract N and make NxN matrix
            p = len(self.covmatrix_str_wedge)
            matrix_dim = int((-1+np.sqrt(1+8*p))/2)
            self.matrix_dim_wedge = matrix_dim

            shape = [matrix_dim, matrix_dim]
            covariance = sp.zeros(*shape)

            # Fill in covariance matrix elements
            for i in range(matrix_dim):
                covariance[i, :] = [expressions_list_wedge[i:matrix_dim+i]]
            self.cov_matrix_wedge = covariance


    def plot_cov(self, var_values, legend = {'Plin': 'black'}, title = 'Covs',
                 xlabel = '$K$ $(h Mpc^{-1})$', ylabel = '$P$ $(h^3 Mpc^{-3})$', output_name = ''):
        """Plot various ingredients of covariance matrix.

        Parameters
        ----------
        var_values : dict
            Dictionary of variable values to use in plots.
        legend : dict
            Dictionary of items to plot, specifying colors for each. Add these terms in variable list!
            Possible keys: Pgg, Pgn, Pnn, shot, Ngg, sh_bis, ...
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

        self.temp_cov_plotting = temp_cov

        ##quicky, but it should be done like above

        spectra = {}

        count = 0
        for k in legend.keys():
            if k in self.covmatrix_str:
                i = int(count/self.cov_matrix_dim)
                j = (count+i)%self.cov_matrix_dim
                spectra[k] = temp_cov[i, j, :]
                count += 1
            else:
                spectra[k] = var_values[k]

        '''
        gg = temp_cov[0, 0, :]
        #spectra['Pgg'] = gg
        try:
            #nn = temp_cov[1, 1, :]
            #gn = temp_cov[0, 1, :]
            #spectra['Pgn'] = gn
            #spectra['Pnn'] = nn
            spectra['sh_bis'] = var_values['sh_bis']
            spectra['sh_tris'] = var_values['sh_tris']
        except:
            a = 1

        if 'Plin' in legend.keys():
            Plin = var_values['Plin']
            spectra['Plin'] = Plin

        spectra['shot'] = K*0.+1./var_values['nhalo']
        spectra['Ngg'] = var_values['Ngg']
        '''

        # Make plot
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale('log')

        for v in legend.keys():
            plt.plot(K, spectra[v], label = v, color = legend[v]['color'], ls = legend[v]['ls'], lw = 2)

        # ax.legend(loc = 'best', prop = {'size': 6})
        ax.legend(loc = 'best')
        # fig.savefig(output_name, dpi = 300)
        fig.savefig(output_name)
        plt.close(fig)


    # def __fab__(self, cgg, dera_cgg, derb_cgg):
    #     A = dera_cgg/cgg
    #     B = derb_cgg/cgg
    #     tot = 1./2.
    #     tot *= A*B
    #     return tot

    def __get__inv(self):
        return 0

    def __getF__(self, covariance_matrix, dera_covariance_matrix, derb_covariance_matrix,
                 var_values = None):
        """Compute Fisher matrix element.

        Parameters
        ----------
        covariance_matrix : array
            Symbolic version of covariance matrix.
        dera_covariance_matrix : array
            Symbolic version of derivative covariance matrix w.r.t. parameter a.
        derb_covariance_matrix : array
            Symbolic version of derivative covariance matrix w.r.t. parameter b.
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
            var_values['po'] = np.ones((self.length_K, self.length_K))

        numpy_covariance_matrix = sp.lambdify(all_vars, covariance_matrix, 'numpy')
        numpy_dera_covariance_matrix = sp.lambdify(all_vars, dera_covariance_matrix+po*sp.ones(*covariance_matrix.shape), 'numpy')
        numpy_derb_covariance_matrix = sp.lambdify(all_vars, derb_covariance_matrix+po*sp.ones(*covariance_matrix.shape), 'numpy')

        # Evaluate covariance and covariance derivatives at input parameter values
        cov_mat = numpy_covariance_matrix(**var_values)
        dera_cov_mat = numpy_dera_covariance_matrix(**var_values)
        derb_cov_mat = numpy_derb_covariance_matrix(**var_values)


        shape = cov_mat.shape

        final = np.zeros((shape[-2], shape[-1]))

        # Compute Fisher matrix
        for i in range(shape[-2]):  #K index
            for j in range(shape[-1]): #mu index
                cov = cov_mat[:, :, i, j]
                dera_cov = dera_cov_mat[:, :, i, j]-var_values['po'][i, j]*1
                derb_cov = derb_cov_mat[:, :, i, j]-var_values['po'][i, j]*1
                invC = np.linalg.inv(cov)
                prod = dera_cov@invC@derb_cov@invC
                final[i, j] += 0.5*np.matrix.trace(prod)

        final = np.array(final)

        return final

    def get_fisher_matrix(self, variables_list = [], var_values = None,
                          verbose = True):
        """Get per-k Fisher matrix, as a matrix of numpy functions.

        Symbolic form deprecated for now.

        Also compute per-k Fischer matrix within foreground wedge, if
        the corresponding covariance matrix exists.

        Parameters
        ----------
        variables_list : list, optional
            List of variable names to include in Fisher matrix. If not specified,
            everything in self.vars is used.
        verbose : bool, optional
            Whether to print some status updates (default: True).
        """
        wedge = self.cov_matrix_wedge is not None

        matrix_dim = self.matrix_dim
        shape = [matrix_dim, matrix_dim]
        if wedge:
            matrix_dim_wedge = self.matrix_dim_wedge
            shape_wedge = [matrix_dim_wedge, matrix_dim_wedge]

        # Compute derivatives of covariance matrix w.r.t. each parameter of interest
        for variable in self.vars:
            der_cov = sp.diff(self.cov_matrix, variable)
            setattr(self, 'der'+str(variable)+'_matrix', der_cov)
            if wedge:
                der_cov = sp.diff(self.cov_matrix_wedge, variable)
                setattr(self, 'der'+str(variable)+'_matrix_wedge', der_cov)

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
        fab_numpy = np.zeros((N, N, self.length_K, self.length_K))
        if wedge:
            fab_numpy_wedge = np.zeros_like(fab_numpy)

        if verbose:
            print('Calculating fisher matrix per mode')

        # For each parameter pair, get Fisher element
        for a, b in combs:
            if verbose:
                print('\t%s, %s' % (a,b))
            dera_cov = self.get_expression('der'+str(a)+'_matrix')
            derb_cov = self.get_expression('der'+str(b)+'_matrix')
            i, j = lista.index(a), lista.index(b)
            f = self.__getF__(self.cov_matrix, dera_cov, derb_cov, var_values)
            fab_numpy[i, j, :, :] = f
            fab_numpy[j, i, :, :] = f
            # Note: fab_numpy is packed like [var, var, mu, k]

            if wedge:
                dera_cov = self.get_expression('der'+str(a)+'_matrix_wedge')
                derb_cov = self.get_expression('der'+str(b)+'_matrix_wedge')
                f = self.__getF__(self.cov_matrix_wedge, dera_cov, derb_cov, var_values)
                fab_numpy_wedge[i, j, :, :] = f
                fab_numpy_wedge[j, i, :, :] = f

        self.fisher = fab_numpy
        self.fisher_numpy = fab_numpy
        if wedge:
            self.fisher_wedge = fab_numpy_wedge
            self.fisher_numpy_wedge = fab_numpy_wedge

        if verbose:
            print('Done calculating fisher matrix per mode')
            if wedge:
                print('\t(Including fisher matrix within foreground wedge)')

    def get_error(self, variable, marginalized = False, integrated = False,
                  kmin = 0.005, kmax = 0.05, volume = 100, Ks = None, recalculate = False,
                  verbose = True, scipy_mode = True, interp_mode = 'cubic',
                  log_integral = False, mu_limit = None,
                  deltag_kmin_kpar = False, add_fg_fisher = False):
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
            Minimum value of long-mode k_min to use in loop over different
            k_min values (default: 0.005).
        kmax : float, optional
            Maximum value of long-mode k_min to use (default: 0.05).
        volume : float, optional
            Survey volume, in (Gpc/h)^3 (default: 100).
        recalculate : bool, optional
            Whether to recompute integrated Fisher matrix if it exists already
            (default: False).
        Ks: array, optional
            Kmin on which to calculate integrated value. (default: None)
            If None gets a value specified by object.
        scipy_mode : bool, optional
            Integrate with scipy.integrate.quad. If False, integrate with
            mpmath.quad. (Default: False).
        interp_mode : string, optional
            Interpolation mode for per-k Fisher matrix. Same options as
            scipy.interpolate.interp1d (default: 'cubic').
        log_integral : bool, optional
            Whether to integrate in k (False) or log(k) (True).
            Default: False.
        mu_limit : float, optional
            If specified, the limit on mu in the Fisher integral corresponding
            to a foreground wedge. (Default: None)
        # different_deltar_kmin : bool, optional
        #     Use kmin as lowest accessible k for delta_r, and loop over
        #     possible k_min values for delta_g. (Default: False)
        deltag_kmin_kpar : bool, optional
            If True, treat deltag_kmin as a bound on k_\parallel in the Fisher
            integral. If False, treat deltag_kmin as a bound on k.
            Default: False
        add_fg_fisher : bool, optional
            If specified, when doing k integral of Fisher matrix, add the
            separate Fisher matrix corresponding to modes within the wedge
            and/or outside of the k or k_\parallel restriction.
            Default: False

        NOTE: option marginalized = True and integrated = False not implemented
            for ill conditioned matrices due to numerical errors
            coming from machine precision.
        """

        if marginalized and not integrated:
            print('Marginalized per mode error not implemented!')
            return 0.

        # If mu_limit is specified, make sure it is between 0 and 1
        if mu_limit is not None:
            if mu_limit > 1. or mu_limit < 0.:
                raise Exception('Must have 0 <= mu_limit <= 1!')

        if add_fg_fisher and self.cov_matrix_wedge is None:
            raise Exception(
                'Extra Fisher matrix must be stored to use add_fg_fisher!'
            )

        no_fg_mode = None
        fg_mode = None
        if mu_limit is not None:
            if deltag_kmin_kpar:
                no_fg_mode = 'outside_wedge_and_kpar_above'
                fg_mode = 'inside_wedge_or_kpar_below'
            else:
                no_fg_mode = 'outside_wedge_and_k_above'
                fg_mode = 'inside_wedge_or_k_below'
        elif mu_limit is None:
            if deltag_kmin_kpar:
                no_fg_mode = 'kpar_above'
                fg_mode = 'kpar_below'
            else:
                no_fg_mode = 'k_above'
                fg_mode = 'k_below'

        # Convert volume from Gpc^3 to Mpc^3
        volume *= 10**9

        # Get unmarginalized error (only used if integrate==False)
        error = self.get_non_marginalized_error_per_mode(variable)

        K = self.K.copy()

        lista = self.fisher_list
        combs = list(itertools.combinations_with_replacement(list(lista), 2))

        if verbose:
            print('List for combinations', lista)

        if integrated:
            # Make array of k_min values to consider. Because of the way
            # the K_min on delta_g is implemented in getIntegratedFisher,
            # Ks cannot be more finely-spaced than K, so let's just use Ks=K
            if Ks is None:
                Ks = K
                # Ks = np.arange(kmin, kmax-0.005, 0.001)
                # Ks = np.geomspace(kmin, kmax-0.001, 50)

            #note: also case where Ks changes
            if self.fisher_integrated is None or recalculate:

                # Define empty array for Fisher values, and fetch parameter pairs
                shape = self.fisher_numpy.shape

                f_int = np.zeros((shape[0], shape[1], len(Ks)))

                if verbose:
                    print('Getting integrated error')
                    if scipy_mode:
                        print('Scipy Integration routine')
                    else:
                        print('Mpmath Integration routine')

                # For each parameter pair, get Fisher element for each k, and
                # do k integral

                for a, b in combs:
                    if verbose:
                        print('\t%s, %s' % (a,b))
                    i, j = lista.index(a), lista.index(b)
                    f = self.fisher_numpy[i, j, ...]
                    if add_fg_fisher:
                        f_wedge = self.fisher_numpy_wedge[i, j, ...]

                    IntegratedFish = np.array([])

                    for Kmin in Ks:
                        error = self.getIntegratedFisher(K, f, kmin, kmax, Kmin,
                                volume, fg_mode=no_fg_mode,
                                scipy_mode=scipy_mode, interp_mode=interp_mode,
                                log_integral=log_integral, mu_limit=mu_limit)

                        if add_fg_fisher:
                            # If desired, compute the integrated Fisher matrix within
                            # the wedge or above kpar_min or k_min,
                            # and add to the non-wedge/high-k result
                            error_fg = self.getIntegratedFisher(
                                K, f_wedge,
                                kmin, kmax, Kmin, volume,
                                fg_mode=fg_mode,
                                scipy_mode=scipy_mode, interp_mode=interp_mode,
                                log_integral=log_integral, mu_limit=mu_limit
                            )
                            error += error_fg

                        IntegratedFish = np.append(IntegratedFish, error)

                    f_int[i, j, :] = IntegratedFish+self.inv_priors[a]*int(i==j)
                    f_int[j, i, :] = f_int[i, j, :]

                self.fisher_integrated = f_int


            if verbose:
                print('Done getting integrated error')

            N = len(lista)

            # Get the errorbar, as either F_{ii}^{-0.5} in the unmarginalized
            # case, or (F^{-1})_{ii}^{0.5} in the marginalized case
            ind1, ind2 = lista.index(variable), lista.index(variable)

            error = self.fisher_integrated[ind1, ind2, :]**-0.5

            if marginalized:
                if self.fisher_integrated_marginalized is None or recalculate:
                    if verbose:
                        print('Marginalizing')
                    # For each k_min, invert the Fisher matrix
                    f_int = self.fisher_integrated
                    cov_int_marg = f_int.copy()
                    for i in range(f_int.shape[-1]):
                        matrix = f_int[:, :, i]
                        try:
                            cov_int_marg[:, :, i] = np.linalg.inv(matrix)
                        except np.linalg.LinAlgError:
                            print('Covariance inversion error! K = %g' % Ks[i])
                            print('Setting F_inv to zero')
                            cov_int_marg[:, :, i] = np.zeros_like(matrix)
                        error_inversion1 = np.max(abs(cov_int_marg[:, :, i]@matrix-np.eye(N)))
                        error_inversion2 = np.max(abs(matrix@cov_int_marg[:, :, i]-np.eye(N)))
                        # or can just check if m*minv = minv*m within some error
                        if (error_inversion1 > 1e-2) or (error_inversion2 > 1e-2):
                            print('WARNING: Fisher matrix inversion not accurate.')
                            print('K:',Ks[i])
                            print('\t max(F^-1 F - I) = %g' % error_inversion1)
                            print('\t max(F F^-1 - I) = %g' % error_inversion2)
                            print('\tF:  ',matrix)
                            print('\tF^-1:',cov_int_marg[..., i])
                    if verbose:
                        print('Done!')
                    self.fisher_integrated_marginalized = cov_int_marg
                error = self.fisher_integrated_marginalized[ind1, ind2, :]**0.5

            K = Ks

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
        error = (self.fisher_numpy[i, j]+self.inv_priors[variable])**-0.5
        return error


    def set_mpmath_integration_precision(self, integration_prec = 53):
        mpmath.mp.prec = integration_prec

    def getIntegratedFisher(self, K, FisherPerMode, kmin, kmax, deltag_kmin, V,
                            fg_mode = 'k_above',
                            apply_filter = False,
                            scipy_mode = True, interp_mode = 'cubic',
                            log_integral = False,
                            mu_limit = None):
        """Integrate per-mode Fisher matrix element in k, to get full Fisher matrix element.

        Given arrays of k values and corresponding Fisher matrix elements F(k), compute
            \frac{V}{(2\pi)^2} \int_{kmin}^{kmax} dk k^2 F(k) ,
        where V is the survey volume. The function actually first interpolates over
        the discrete supplied values of FisherPerMode, and then integrates the
        interpolating function between kmin and kmax.

        Optional arguments control whether to only perform this integral outside or
        inside of a 21cm foreground wedge specified by mu_limit, whether to
        restruct k or k_\parallel to be above or below a certain value, or some
        combination of those two conditions.

        Parameters
        ----------
        K : ndarray
            Array of k values.
        FisherPerMode : array
            Array of Fisher element values corresponding to [mu,K].
        kmin : float
            Lowest k accessible in survey. Sets lower limit of numerical integral,
            but actual lower limit is determined by deltag_kmin.
        kmax : float
            Upper limit of k integral.
        deltag_kmin : float, optional
            By default, this sets the effective lower limit of the k integral,
            either as an isotropic limit on |\vec{k}|, or as a limit on
            k_\parallel, depending on the value of fg_mode.
        V : float
            Survey volume.
        fg_mode : ['outside_wedge', 'inside_wedge',
                   'k_above', 'k_below',
                   'kpar_above', 'kpar_below',
                   'kpar_kperp_above', 'kpar_kperp_below',
                   'outside_wedge_and_kpar_above', 'inside_wedge_or_kpar_below',
                   'outside_wedge_and_k_above', 'inside_wedge_or_k_below',
                   'outside_wedge_and_kpar_kperp_above',
                   'inside_wedge_or_kpar_kperp_below'],
                   optional
            Restrict k and mu integrals of Fisher matrix according to various
            options. Default: 'k_above'
        apply_filter : bool, optional
            Smooth the per-k Fisher matrix before integration (default: False).
        scipy_mode : bool, optional
            Integrate with scipy.integrate.quad. If False, integrate with
            mpmath.quad. (Default: False).
        interp_mode : string, optional
            Interpolation mode for per-k Fisher matrix. Same options as
            scipy.interpolate.interp1d (default: 'cubic').
        log_integral : bool, optional
            Whether to integrate in k (False) or log(k) (True).
            Default: False.
        mu_limit : float, optional
            If specified, when doing k integral of Fisher matrix, only integrate
            mu in within [-1, -mu_limit] and [mu_limit, 1] , or within
            [-mu_limit, mu_limit], depending on fg_mode (see below). Default: None
        kpar_limit : float, optional
            Minimum or maximum k_\parallel considered in Fisher integral,
            in case where deltag_min corresponds to k_\perp. kpar_limit is
            ignored unless fg_mode in ['kpar_kperp_above',
            'kpar_kperp_below', 'outside_wedge_and_kpar_kperp_above',
            'inside_wedge_or_kpar_kperp_below']. Default: None

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

        fg_mode_options = ['outside_wedge', 'inside_wedge',
                   'k_above', 'k_below',
                   'kpar_above', 'kpar_below',
                   'kpar_kperp_above', 'kpar_kperp_below',
                   'outside_wedge_and_kpar_above', 'inside_wedge_or_kpar_below',
                   'outside_wedge_and_k_above', 'inside_wedge_or_k_below',
                   'outside_wedge_and_kpar_kperp_above',
                   'inside_wedge_or_kpar_kperp_below']
        if fg_mode is not None and fg_mode not in fg_mode_options:
            raise Exception('Invalid fg_mode specified!')

        if fg_mode in ['kpar_kperp_above', 'kpar_kperp_below',
                   'outside_wedge_and_kpar_kperp_above',
                   'inside_wedge_or_kpar_kperp_below'] and kpar_limit is None:
            raise Exception(
                'If fg_mode is %s, kpar_limit must be specified!' % fg_mode
            )

        if apply_filter:
            lenK = len(K)
            window_length = 7 #2*int(lenK/10)+1 #make sure it is an odd number
            FisherPerMode = savgol_filter(FisherPerMode, window_length, 3) #, mode = 'nearest')

        muInt_factor = np.ones_like(K)

        if fg_mode == 'outside_wedge':
            muInt_factor *= (1.-mu_limit)

        elif fg_mode == 'inside_wedge':
            muInt_factor *= mu_limit

        elif fg_mode == 'k_above':
            muInt_factor *= (K >= deltag_kmin)

        elif fg_mode == 'k_below':
            muInt_factor *= (K < deltag_kmin)

        elif fg_mode == 'kpar_above':
            muInt_factor *= (1. - np.array([min(1.,x) for x in deltag_kmin/K]))

        elif fg_mode == 'kpar_below':
            muInt_factor *= np.array([min(1.,x) for x in deltag_kmin/K])

        elif fg_mode == 'kpar_kperp_above':
            raise NotImplementedError('fg_mode=kpar_kperp_above is not implemented!')
            # muInt_factor *= (1. - np.array([min(1.,x) for x in deltag_kmin/K]))

        elif fg_mode == 'kpar_kperp_below':
            raise NotImplementedError('fg_mode=kpar_kperp_below is not implemented!')
            # muInt_factor *= np.array([min(1.,x) for x in deltag_kmin/K])

        elif fg_mode == 'outside_wedge_and_kpar_above':
            muInt_factor *= (1. - np.array([max(min(1.,x), mu_limit) for x in deltag_kmin/K]))

        elif fg_mode == 'inside_wedge_or_kpar_below':
            muInt_factor *= np.array([max(min(1.,x), mu_limit) for x in deltag_kmin/K])

        elif fg_mode == 'outside_wedge_and_k_above':
            muInt_factor *= (1.-mu_limit)
            muInt_factor *= (K >= deltag_kmin)

        elif fg_mode == 'inside_wedge_or_k_below':
            muInt_factor *= (1.-mu_limit)
            muInt_factor *= (K >= deltag_kmin)
            muInt_factor = 1. - muInt_factor

        FisherPerModeLocal = FisherPerMode.copy()
        # Since FisherPerMode is packed as [par, par, mu, K],
        # taking FisherPerModeLocal *= muInt_factor will multiply by
        # muInt_factor along the K axis, as desired
        FisherPerModeLocal *= muInt_factor

        # function = scipy.interpolate.interp1d(K, FisherPerModeLocal,
        #                                       kind=interp_mode,
        #                                       bounds_error=False, fill_value=0)
        mus = np.linspace(-1, 1, len(K))

        function = scipy.interpolate.interp2d(K, mus, FisherPerModeLocal, interp_mode,
                                                fill_value = 0., bounds_error = 0.)
        f = lambda x1,x2 : x1**2.*(si.dfitpack.bispeu(function.tck[0], function.tck[1],
            function.tck[2], function.tck[3], function.tck[4], x1, x2)[0])[0]

        #################
        # kmin = deltag_kmin
        ##################

        if scipy_mode:
            if log_integral:
                # result = scipy.integrate.quad(lambda L: function(np.exp(L))*(np.exp(L))**3.,
                #                                 np.log(kmin), np.log(kmax), epsrel = 1e-15)
                result = scipy.integrate.dblquad(
                    lambda y, L: function(np.exp(L), y)*(np.exp(L))**3.,
                    np.log(kmin),
                    np.log(kmax),
                    lambda x: -1.,
                    lambda x: 1.,
                    epsrel = 1e-15
                )
            else:
                # result = scipy.integrate.quad(lambda x: function(x)*x**2.,
                #                                 kmin, kmax, epsrel = 1e-15)
                result = scipy.integrate.dblquad(
                    lambda y, x: function(x, y)*x**2.,
                    kmin,
                    kmax,
                    lambda x: -1.,
                    lambda x: 1.,
                    epsrel = 1e-15
                )
        else:
            ## A bit slow but it is worth it for numerical precision
            if log_integral:
                # def f(L):
                #     L = float(L)
                #     y = function(np.exp(L))*np.exp(L)**3.
                #     return mpmath.mpf(1)*y
                #
                # resultmp = mpmath.quad(f, [np.log(kmin), np.log(kmax)])
                # result = [resultmp]
                raise NotImplementedError('log_integral not implemented for mpmath!')
            else:
                # def f(x):
                #     x = float(x)
                #     y = function(x)*x**2.
                #     return mpmath.mpf(1)*y
                #
                # resultmp = mpmath.quad(f, [kmin, kmax])
                # result = [resultmp]
                resultmp = mpmath.quad(f, [kmin, kmax], [-1, 1])
                result = [resultmp]

        # result = result[0]*V/(4.*np.pi**2.)
        result = result[0]*V/(2.*np.pi)**2.
        return result

    def plot_forecast(self, variable, error_versions, scipy_mode = True, kmin = 0.005, kmax = 0.05,
                     volume = 100, title = 'Error', xlabel = '$K$ $(h Mpc^{-1})$',
                     ylabel = '$\sigma$', xscale = 'linear', yscale = 'log', output_name = '',
                     style = 'default', rescale_y=1):

        plt.style.use(style)
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.grid(which = 'both')

        for label, value in error_versions.items():
            K, error = self.get_error(variable, value['marginalized'], value['integrated'],
                                      kmin, kmax, volume, scipy_mode = scipy_mode)
            plt.plot(K, rescale_y * error, label = label, lw = 2)
        ax.legend(loc = 'best')
        # ax.legend(loc = 'best', prop = {'size': 6})
        # fig.savefig(output_name, dpi = 300)
        fig.savefig(output_name)
        plt.close(fig)
