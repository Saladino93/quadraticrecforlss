import sympy as sp

import numpy as np

import itertools

import matplotlib.pyplot as plt

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

    def __init__(self, K, *args):
        self.K = K
        self.length_K = len(K)
        expression.__init__(self, *args)
        self.K = K
        self.length_K = len(K)

    def add_cov_matrix(self, covariance_matrix_dict, ):
        elements = []
        expressions_list = []
        for key, value in covariance_matrix_dict.items():
            self.add_expression(key, value)
            expressions_list += [self.get_expression(key)]
            elements += [key]
        self.covmatrix_str = elements

        p = len(self.covmatrix_str)
        matrix_dim = int((-1+np.sqrt(1+8*p))/2)
        self.matrix_dim = matrix_dim

        shape = [matrix_dim, matrix_dim]
        covariance = sp.zeros(*shape)

        for i in range(matrix_dim):
            covariance[i, :] = [expressions_list[i:matrix_dim+i]] 
        #print(expressions_list) 
        #print(covariance)
        #symm = covariance+covariance.T
        #covariance = sp.Matrix(matrix_dim, matrix_dim, lambda i, j: symm[i, j]/2 if i==j else symm[i, j])
        #print(covariance) 
        self.cov_matrix = covariance

 
    def plot_cov(self, var_values, legend = {'Plin': 'black'}, title = 'Covs', xlabel = '$K$ $(h Mpc^{-1})$', ylabel = '$P$ $(h^3 Mpc^{-3})$', output_name = ''):

        K = self.K

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
          

        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        for v in legend.keys():
            plt.plot(K, spectra[v], label = v, color = legend[v]['color'], ls = legend[v]['ls'])


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
    def __getF__(self, covariance_matrix, dera_covariance_matrix, derb_covariance_matrix, numpify = False, var_values = None):
        if numpify:
                all_vars = self.vars

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
                
                cov_mat = numpy_covariance_matrix(**var_values)
                dera_cov_mat = numpy_dera_covariance_matrix(**var_values)
                derb_cov_mat = numpy_derb_covariance_matrix(**var_values)

                shape = cov_mat.shape

                final = []

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

        for variable in self.vars:
            der_cov = sp.diff(self.cov_matrix, variable) 
            setattr(self, 'der'+str(variable)+'_matrix', der_cov)
    
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

        for a, b in combs:
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

   
    def get_non_marginalized_error(self, variable):
        lista = self.fisher_list
        i, j = lista.index(variable), lista.index(variable) 
        error = (self.fisher_numpy[i, j])**-0.5
        return error       


    def get_marginalized_error(self, variable):
        lista = self.fisher_list
        i, j = lista.index(variable), lista.index(variable)
        error = (self.fisher_numpy[i, j])**-0.5
        return error 


    def get_integrated_error(self, variable, marginalized = False, kmin = 0.005, kmax = 0.05, volume = 100):
        if not marginalized:
            getIntregratedFisher(self.K, self.get_non_marginalized_error(variable), kmin, kmax, volume)     
        return 0.

    def getIntregratedFisher(self, K, FisherPerMode, kmin, kmax, V):
        if (kmin<np.min(K)) or (kmax>np.max(K)):
            print('Kmin(Kmax) should be higher(lower) than the minimum(maximum) of the K avaliable!')
            return 0
        else:
            function = scipy.interpolate.interp1d(K, FisherPerMode)
            result = scipy.integrate.quad(lambda x: function(x)*x**2., kmin, kmax)
            result = result[0]*V/(2.*np.pi**2.)
            return result

    def plot_forecast(self, variable, marginalized = False, integrated = False, title = 'Error', xlabel = '$K$ $(h Mpc^{-1})$', ylabel = '\sigma', output_name = ''):
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if not marginalized:
            error = self.get_non_marginalized_error(variable)
        plt.plot(self.K, error)
        ax.legend(loc = 'best', prop = {'size': 6})
        fig.savefig(output_name, dpi = 300)
        plt.close(fig)
