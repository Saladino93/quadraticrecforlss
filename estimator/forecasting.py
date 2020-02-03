"""Computes various Fisher matrices using directly-measured and reconstructed fields in LSS.
"""

import sympy as sp
import numpy as np
import itertools

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

    def __init__(self, *args):
        """
        Parameters
        ----------
        args : list
            List of variables we ultimately want to forecast for.
        """
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

    def __fab__(self, cgg, dera_cgg, derb_cgg):
        A = dera_cgg/cgg
        B = derb_cgg/cgg
        tot = 1./2.
        tot *= A*B
        return tot

    def __getF__(self, covariance_matrix, dera_covariance_matrix, derb_covariance_matrix):
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
        final = 0.5*tr #0.5*sp.simplify(tr)
        return final


    def get_fisher_matrix(self, variables_list = [], numpify = False, verbose = True):
        """Get per-k Fisher matrix, both in symbolic form and as a matrix of numpy functions.

        Parameters
        ----------
        variables_list : list, optional
            List of variable names to include in Fisher matrix. If not specified,
            everything in self.vars is used.
        numpyify : bool, optional
            Whether to generate numpy-function version of Fisher matrix (default: False).
        verbose : bool, optional
            Whether to print some status updates (default: True).
        """
        matrix_dim = self.matrix_dim
        shape = [matrix_dim, matrix_dim]

        for variable in self.vars:
            der_cov = sp.zeros(1, matrix_dim**2)
            for expression in self.covmatrix_str:
                i = self.covmatrix_str.index(expression)
                der_cov[0, i] = self.derivate(expression, variable, save_derivative = True)

            der_cov = der_cov.reshape(*shape)

            for i in range(matrix_dim):
                for j in range(i, matrix_dim):
                    der_cov[j, i] = der_cov[i, j]


            setattr(self, 'der'+str(variable)+'_matrix', der_cov)

        if variables_list == []:
            lista = self.vars
            s = ' all '
        else:
            lista = variables_list
            s = ' '

        if verbose:
            print('Using'+s+'parameters list:', lista)

        combs = list(itertools.combinations_with_replacement(list(lista), 2))

        N = len(lista)
        shape = [N, N]
        fab = sp.zeros(*shape)

        fab_numpy = {}

        if verbose:
            print('Calculating fisher matrix')

        for a, b in combs:
            dera_cov = self.get_expression('der'+str(a)+'_matrix')
            derb_cov = self.get_expression('der'+str(b)+'_matrix')
            i, j = lista.index(a), lista.index(b)
            fab[i, j] = self.__getF__(self.cov_matrix, dera_cov, derb_cov)

            if numpify:
                all_vars = self.vars
                numpyified = sp.lambdify(all_vars, fab[i, j], 'numpy')
                fab_numpy[str(a)+''+str(b)] = numpyified
                fab_numpy[str(b)+''+str(a)] = numpyified
            #i, j = lista.index(a), lista.index(b)
            #fab[i, j] = self.__fab__(cgg, dera_cov, derb_cov)

        self.fisher = fab
        self.fisher_numpy = fab_numpy


    def get_non_marginalized_error(self, variable, **inputs):
        error = (self.fisher_numpy[variable+variable](**inputs))**-0.5
        return error
