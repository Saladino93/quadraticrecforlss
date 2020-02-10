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
       
dic['fnl'] = 0.
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

###### FORECAST ######

forecast = forecasting.Forecaster(K, *variables_list)

#here take biases definitions and convert them to sympy
for x in terms:
    exec(x+'=0')
    globals()[x] = sp.sympify(biases_definitions[x], locals = forecast.ns)
    forecast.ns[x] = globals()[x]
   

#define noise variables
for x, y in combs:
    exec(noise_prefix+x+y+'=0')
    globals()[noise_prefix+x+y] = sp.symbols(noise_prefix+x+y)
    exec(noise_prefix+y+x+'=0')
    globals()[noise_prefix+y+x] = sp.symbols(noise_prefix+y+x)     
    
    forecast.ns[noise_prefix+x+y] = globals()[noise_prefix+x+y]
    forecast.ns[noise_prefix+y+x] = globals()[noise_prefix+y+x] 

#here take new bias of the reconstructed field
forecast.new_bias = sp.sympify(new_bias_expr, locals = forecast.ns)
forecast.ns['new_bias'] = sp.sympify(new_bias_expr, locals = forecast.ns)


forecast.add_cov_matrix(cov_dict)
forecast.plot_cov(var_values, legend = legend_cov, title = title_cov, xlabel = xlabel_cov, ylabel = ylabel_cov, output_name = direc+pics_dir+output_name_cov+'.png')


forecast.get_fisher_matrix(variables_list_fisher, var_values = var_values)

forecast.set_mpmath_integration_precision(100)

error_versions = {'Per mode not integrated ': {'marginalized': False, 'integrated': False}, 'Integrated marginalized': {'marginalized': False, 'integrated': True}}
for v in variables_of_interest:
    forecast.plot_forecast(v, error_versions, kmin = K.min(), kmax = K.max(), volume = 100, xlabel = xlabel, ylabel = ylabel, xscale = xscale, yscale = yscale, output_name = direc+pics_dir+output_name+v+'.png')



kf,sig_fnl = forecast.get_error('fnl', marginalized = False, integrated = True,
              kmin = K.min(), kmax = K.max(),
              volume = values['survey_config']['geometry']['volume'])
np.savetxt(direc+data_dir+'sigma_fnl_unmarg_int.dat',np.c_[kf, sig_fnl])

kf, sig_fnl_marg_int = forecast.get_error('fnl', marginalized = True, integrated = True,
              kmin = K.min(), kmax = K.max(),
              volume = values['survey_config']['geometry']['volume'])
np.savetxt(direc+data_dir+'sigma_fnl_marg_int.dat',np.c_[kf, sig_fnl_marg_int])








