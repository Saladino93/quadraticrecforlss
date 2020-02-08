import sympy as sp

import numpy as np

import itertools

#IDEA: to combine forecasts just create a + operation for the class

class expression():
    # args here is just the list of variables used 
    def __init__(self, *args):
        var_list = []
        var_list_names = []
        for a in args:
            symb = sp.symbols(a)
            setattr(self, a, symb)
            var_list += [symb]
            var_list_names += [a]
        self.vars = var_list
        self.vars_names = var_list_names

    def add_extra_var(self, name, expression):
        setattr(self, name, expression)

    # similar to add_extra_var but logic of usage is different  
    def add_expression(self, expression_name: str, expr):
        setattr(self, expression_name, sp.sympify(expr))
        
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
        setattr(self, expression_name, sp.sympify(new_expression_value))
        return


class Forecaster(expression):

    def __init__(self, *args):
        expression.__init__(self, *args)

    def add_cov_matrix(self, covariance_matrix_dict):
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
            covariance[i, i:] = [expressions_list[:matrix_dim-i]] 
            
        symm = covariance+covariance.T
        covariance = sp.Matrix(matrix_dim, matrix_dim, lambda i, j: symm[i, j]/2 if i==j else symm[i, j])
       
        self.cov_matrix = covariance

    def __fab__(self, cgg, dera_cgg, derb_cgg):
        A = dera_cgg/cgg
        B = derb_cgg/cgg
        tot = 1./2.
        tot *= A*B
        return tot

    def __getF__(self, covariance_matrix, dera_covariance_matrix, derb_covariance_matrix):
        invC = covariance_matrix.inv()
        prod = dera_covariance_matrix*invC*derb_covariance_matrix*invC
        tr = prod.trace()
        final = 0.5*sp.simplify(tr)
        return final 


    def get_fisher_matrix(self, variables_list = [], numpify = False, verbose = True):

        matrix_dim = self.matrix_dim
        shape = [matrix_dim, matrix_dim]

        for variable in expr.vars:
            der_cov = sp.zeros(1, matrix_dim**2)
            for expression in self.covmatrix_str: 
                i = self.covmatrix_str.index(expression)
                der_cov[0, i] = expr.derivate(expression, variable, save_derivative = True) 
          
            der_cov = der_cov.reshape(*shape)

            for i in range(matrix_dim):
                for j in range(i, matrix_dim):
                    der_cov[j, i] = der_cov[i, j]


            setattr(self, 'der'+str(variable)+'_matrix', der_cov)
    
        if variables_list == []:
            lista = expr.vars
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
            cgg = self.get_expression(expression)
            i, j = lista.index(a), lista.index(b)
            fab[i, j] = self.__getF__(self.cov_matrix, dera_cov, derb_cov)
            if numpify:
                all_vars = expr.vars
                numpyified = sp.lambdify(all_vars, fab[i, j], 'numpy')
                fab_numpy[str(a)+''+str(b)] = numpyified
                fab_numpy[str(b)+''+str(a)] = numpyified

            #i, j = lista.index(a), lista.index(b)
            #fab[i, j] = self.__fab__(cgg, dera_cov, derb_cov)
        
        
        self.fisher_numpy = fab_numpy
    

    def get_non_marginalized_error(self, variable, **inputs):
        error = (self.fisher_numpy[variable+variable](**inputs))**-0.5
        return error         

variables_list = ['b', 'fnl', 'nbar', 'Pnl', 'func', 'b2']

expr = Forecaster(*variables_list)

#List for an upper triangular matrix
#You have to imagine to reshape it to a square SYMMETRIC matrix
cov_dict = {'Pgg': 'b**2*Pnl+1/nbar', 'Pgn': 'b*b2*Pnl', 'Pnn': 'b2**2*Pnl*fnl+1/nbar'}

expr.add_cov_matrix(cov_dict)

variables_list_fisher = ['b', 'fnl', 'nbar', 'b2']
numpify =  True

expr.get_fisher_matrix(variables_list_fisher, numpify = numpify)

dictionary = {'b': 1, 'fnl': 1, 'nbar': 1, 'Pnl': np.ones(10), 'func': np.ones(10), 'b2': 1}

bb_err = expr.get_non_marginalized_error('b', **dictionary)

print(bb_err)

'''
expr.add_extra_var('bias', expr.b+expr.fnl*expr.func)

expr.add_expression('P_tot', expr.bias**2.*expr.Pnl+1/expr.nbar)

print(expr.P_tot)

for variable in expr.vars:
    expr.derivate('P_tot', variable, save_derivative = True)
    print('der'+str(variable), getattr(expr, 'der'+str(variable)+'_'+'P_tot'))

dictionary = {'b': 1, 'fnl': 1, 'nbar': 1, 'Pnl': np.ones(10), 'func': np.ones(10), 'b2': 1}
print(expr.evaluate_expression.__code__.co_varnames)
print(expr.evaluate_expression('P_tot', **dictionary))
'''
