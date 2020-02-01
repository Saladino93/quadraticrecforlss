from quadforlss import forecasting

import yaml

import pickle

import sys

import sympy as sp

import numpy as np

import itertools

if len(sys.argv) == 1:
    print('Choose your configuration file!')
    sys.exit()

## Read configuration file

values_file = str(sys.argv[1])

with open(values_file, 'r') as stream:
    data = yaml.safe_load(stream)

values = data


direc = values['name']
base_dir = values['file_config']['base_dir']
data_dir = values['file_config']['data_dir']
pics_dir = values['file_config']['pics_dir']
dic_name = values['file_config']['dic_name']

direc = base_dir+direc+'/'

with open(direc+data_dir+dic_name, 'rb') as handle:
    dic = pickle.load(handle, encoding = 'latin1')


config = values['forecast_config']

variables_list= config['variables_list']
variables_list_fisher = config['variables_list_fisher']
cov_dict = config['cov_dict']

variables_of_interest = config['variables_of_interest']

analysis_config = values['analysis_config']
biases_definitions = analysis_config['biases_definitions']
noise_prefix = analysis_config['noise_prefix']
new_bias_expr = analysis_config['new_bias_expr']

pics_config = config['pics_config']
for key, val in pics_config.items():
    exec(key + '=val')
       

terms = biases_definitions.keys()
combs = list(itertools.combinations_with_replacement(list(terms), 2))

Noisedic = {}

#LOAD VARIABLES VALUES
for a, b in combs:
    Noisedic[noise_prefix+a+b] = dic[noise_prefix+a+b]
    Noisedic[noise_prefix+b+a] = dic[noise_prefix+a+b]

K = dic['K']

var_values = {}

for vv in variables_list:
    var_values[vv] = dic[vv]

print(var_values)

###### FORECAST ######

forecast = forecasting.Forecaster(K, *variables_list)

#NOTE#maybe just use ns of forecast

'''
#Here define basic variables, converting them to sympy variables
for v in variables_list:
     exec(v+'=0')
     globals()[v] = sp.symbols(v)
     ns[v] = globals()[v]
'''    

#here take biases definitions and convert them to sympy
for x in terms:
    exec(x+'=0')
    globals()[x] = sp.sympify(biases_definitions[x], locals = forecast.ns)
    forecast.ns[x] = globals()[x]
   

#for y in terms:
#    print(y)
#    print(globals()[y])     

#define noise variables
for x, y in combs:
    exec(noise_prefix+x+y+'=0')
    globals()[noise_prefix+x+y] = sp.symbols(noise_prefix+x+y)
    exec(noise_prefix+y+x+'=0')
    globals()[noise_prefix+y+x] = sp.symbols(noise_prefix+y+x)     
    
    forecast.ns[noise_prefix+x+y] = globals()[noise_prefix+x+y]
    forecast.ns[noise_prefix+y+x] = globals()[noise_prefix+y+x] 

    #forecast.vars += [globals()[noise_prefix+x+y]]
    #if x != y:
    #    forecast.vars += [globals()[noise_prefix+y+x]]

numpify =  True

#forecast = forecasting.Forecaster(*variables_list)

#here take new bias of the reconstructed field
forecast.new_bias = sp.sympify(new_bias_expr, locals = forecast.ns)
forecast.ns['new_bias'] = sp.sympify(new_bias_expr, locals = forecast.ns)

#var_values['new_bias'] = 

forecast.add_cov_matrix(cov_dict)
forecast.plot_cov(var_values, legend = legend_cov, title = title_cov, xlabel = xlabel_cov, ylabel = ylabel_cov, output_name = direc+pics_dir+output_name_cov+'.png')


del forecast.ns['new_bias']
var_values['new_bias'] = 10000

##TO CHECK: no dependence on new_bias var

forecast.get_fisher_matrix(variables_list_fisher, numpify = numpify, var_values = var_values)

#can also loop over all other variables of fisher list
#could put a dictionary for labels
for v in variables_of_interest:
    forecast.plot_forecast(v, marginalized = False, integrated = False, title = title, xlabel = xlabel, ylabel = ylabel, output_name = direc+pics_dir+output_name+v+'.png')
    
