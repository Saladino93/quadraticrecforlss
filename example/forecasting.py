import sympy as sp


#what happens if I create another object and same var names but I want different values? Does sympy create two different instances?

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
    def add_expression(self, expression_name, expr):
        setattr(self, expression_name, sp.sympify(expr))
        
    def __add__(self, other_object_expression):
        return  

    def evaluate_expression(self, expression_name, **valuesdict):
        expr = getattr(self, expression_name)
        expr_numpy = sp.lambdify(self.vars, expr, 'numpy')
        return expr_numpy(**valuesdict)

    def derivate(self, expression_name, variable_of_derivation):
        return sp.diff(getattr(self, expression_name), variable_of_derivation)    

    def change_expression(self, expression_name, new_expression_value):
        return

expr = expression('b', 'fnl', 'nbar', 'Pnl', 'func')
expr.add_extra_var('bias', expr.b+expr.fnl*expr.func)

expr.add_expression('P_tot', expr.bias**2.+1/expr.nbar)

print(expr.P_tot)
print(expr.derivate('P_tot', expr.b))

dictionary = {'b': 1, 'fnl': 1, 'nbar': 1, 'Pnl': 1, 'func': 1}
print(expr.evaluate_expression.__code__.co_varnames)
print(expr.evaluate_expression('P_tot', **dictionary))
