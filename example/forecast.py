"""Compute Fisher forecast for f_NL.
"""

from quadforlss import forecasting

import yaml

import pickle

import sys

import sympy as sp

import numpy as np

import itertools

import matplotlib.pyplot as plt


#################################
# Read input values, spectra, and N_{ab} matrix from disk
#################################

# Read name of config file from command line
if len(sys.argv) == 1:
    print('Choose your configuration file!')
    sys.exit()

# Read config options from yaml file
values_file = str(sys.argv[1])
with open(values_file, 'r') as stream:
    data = yaml.safe_load(stream)
values = data

# Get directory and file names
direc = values['name']
base_dir = values['file_config']['base_dir']
data_dir = values['file_config']['data_dir']
pics_dir = values['file_config']['pics_dir']
dic_name = values['file_config']['dic_name']

direc = base_dir+direc+'/'

# Read dictionary from pickle file
with open(direc+data_dir+dic_name, 'rb') as handle:
    dic = pickle.load(handle, encoding = 'latin1')

# Get lists of variables: all defined variables, variables varying in Fisher
# matrix, dict of auto and cross spectra for data covariance matrix, and
# and list of variables we want to forecast for
config = values['forecast_config']

variables_list= config['variables_list']
variables_list_fisher = config['variables_list_fisher']
cov_dict = config['cov_dict']

variables_of_interest = config['variables_of_interest']

# Get expressions defining different bias coefficients, string prefix
# denoting noise, and multiplicative bias value for new field
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

try:
    var_values['fnlScaling'] = float(config['fnlScaling'])
    fnlScaling = var_values['fnlScaling']
    var_values['fnl'] /= fnlScaling
except KeyError:
    fnlScaling = 1

###### FORECAST ######

# Define Forecaster object
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

#here take new bias of the reconstructed field
forecast.new_bias = sp.sympify(new_bias_expr, locals = forecast.ns)
forecast.ns['new_bias'] = sp.sympify(new_bias_expr, locals = forecast.ns)

#var_values['new_bias'] =

forecast.add_cov_matrix(cov_dict)

legend_cov = {'Plin': {'color': 'black', 'ls': '-'},
              'Pgg': {'color': 'red', 'ls': '-'},
              'shot': {'color': 'blue', 'ls': '-'},
              'Ngg': {'color': 'green', 'ls': '-'}}
output_name_cov = 'test_cov'

# forecast.plot_cov(var_values, legend = legend_cov, title = 'Covariance elements',
#                   output_name = direc+pics_dir+output_name_cov+'.pdf')

forecast.get_fisher_matrix(variables_list_fisher, numpify=True,
                            var_values = var_values)

# print(forecast.cov_matrix[0,1])
print(forecast.K)
# print(forecast.fisher_numpy[:,:,-5])
print(forecast.getIntregratedFisher(forecast.K, forecast.fisher_numpy[0,0,:], 0.045,0.05, 1e9*values['survey_config']['geometry']['volume']))
print(forecast.getIntregratedFisher(forecast.K, forecast.fisher_numpy[0,1,:], 0.045,0.05, 1e9*values['survey_config']['geometry']['volume']))
print(forecast.getIntregratedFisher(forecast.K, forecast.fisher_numpy[1,1,:], 0.045,0.05, 1e9*values['survey_config']['geometry']['volume']))
# for i in range(len(variables_list_fisher)):
#     for j in range(i,len(variables_list_fisher)):
#         plt.plot(forecast.K,forecast.fisher_numpy[i,j,:],label='%d,%d' % (i,j))
# plt.xscale('log')
# plt.legend()
# plt.savefig('/Users/sforeman/Desktop/fish.pdf')

kf,sig_fnl = forecast.get_error('fnl', marginalized = False, integrated = False,
              kmin = K.min(), kmax = K.max(),
              volume = values['survey_config']['geometry']['volume'])
np.savetxt(direc+data_dir+'sigma_fnl_unmarg_perk.dat',np.array((kf,sig_fnl*fnlScaling)).T)

kf,sig_fnl = forecast.get_error('fnl', marginalized = False, integrated = True,
              kmin = K.min(), kmax = K.max(),
              volume = values['survey_config']['geometry']['volume'])
np.savetxt(direc+data_dir+'sigma_fnl_unmarg_int.dat',np.array((kf,sig_fnl*fnlScaling)).T)

kf,sig_fnl = forecast.get_error('fnl', marginalized = True, integrated = True,
              kmin = K.min(), kmax = K.max(),
              volume = values['survey_config']['geometry']['volume'])
np.savetxt(direc+data_dir+'sigma_fnl_marg_int.dat',np.array((kf,sig_fnl*fnlScaling)).T)

error_versions = {
    'Non-marg, non-integrated': {'marginalized': False, 'integrated': False},
    'Non-marg, integrated': {'marginalized': False, 'integrated': True},
    'Marg, integrated': {'marginalized': True, 'integrated': True}
}
forecast.plot_forecast('fnl', error_versions,
                        kmin = K.min(), kmax = K.max(),
                        volume = values['survey_config']['geometry']['volume'],
                        xlabel = r'$k_{\rm min}\; [h\, {\rm Mpc}^{-1}]$',
                        output_name = direc+pics_dir+output_name+'test_forecast.pdf',
                        rescale_y = fnlScaling)



# del forecast.ns['new_bias']
# var_values['new_bias'] = 10000

##TO CHECK: no dependence on new_bias var

# forecast.get_fisher_matrix(variables_list_fisher, numpify = numpify, var_values = var_values)

########print(forecast.get_marginalized_error_per_mode(variables_of_interest[0]))

# #can also loop over all other variables of fisher list
# #could put a dictionary for labels
# error_versions = {'a': {'marginalized': False, 'integrated': False}, 'b': {'marginalized': False, 'integrated': True}}
# for v in variables_of_interest:
#     forecast.plot_forecast(v, error_versions, kmin = K.min(), kmax = K.max(), volume = 100, xlabel = xlabel, ylabel = ylabel, xscale = xscale, yscale = yscale, output_name = direc+pics_dir+output_name+v+'.png')
