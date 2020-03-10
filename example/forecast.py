"""Compute Fisher forecast for f_NL.
"""

from quadforlss import forecasting

import yaml

import pickle

import sys

import sympy as sp

import numpy as np

from scipy.interpolate import interp1d

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
priors = config['priors']
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

# dic['fnl'] = 0.

terms = biases_definitions.keys()
combs = list(itertools.combinations_with_replacement(list(terms), 2))
'''
Noisedic = {}

#LOAD VARIABLES VALUES
for a, b in combs:
    Noisedic[noise_prefix+a+b] = dic[noise_prefix+a+b]
    Noisedic[noise_prefix+b+a] = dic[noise_prefix+a+b]
'''

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

# If file containing N_ab scaling fractions (as an effective way of
# accounting for a foreground wedge) is specified, import it and create
# an interpolating function in k.
if 'Nab_wedge_scaling_fractions' in config.keys():
    print('Scaling N_ab curves to account for lost modes due to foreground wedge')
    Nab_wedge_frac_data = np.loadtxt(config['Nab_wedge_scaling_fractions']).T
    Nab_wedge_fraction = interp1d(Nab_wedge_frac_data[0], Nab_wedge_frac_data[1])

# For each stored N_ab curve, divide it by the N_ab scaling fraction at
# the specified k. This divides the noise curves by the fraction of modes that
# are outside of the wedge, which should be a decent approximation for the
# effect of the modes that are lost to the wedge.
for vv in var_values.keys():
    if vv[0] == "N":
        print('Rescaling %s' % vv)
        var_values[vv] /= Nab_wedge_fraction(K)

# If a mu_limit is specified, grab it
mu_limit = None
if 'fisher_mu_limit' in config.keys():
    mu_limit = config['fisher_mu_limit']
    print('Fisher mu limit found: %g' % mu_limit)


###### FORECAST ######

# Define Forecaster object
forecast = forecasting.Forecaster(K, priors, *variables_list)

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

legend_cov = {'Plin': {'color': 'black', 'ls': '-'},
              'Pgg': {'color': 'red', 'ls': '-'},
              'shot': {'color': 'blue', 'ls': '-'},
              'Ngg': {'color': 'green', 'ls': '-'}}
output_name_cov = 'test_cov'

forecast.plot_cov(var_values, legend = legend_cov, title = 'Covariance elements',
                  output_name = direc+pics_dir+output_name_cov+'.pdf')

forecast.get_fisher_matrix(variables_list_fisher, var_values = var_values)

forecast.set_mpmath_integration_precision(100)

# print(forecast.cov_matrix[0,1])
# print(forecast.K)
# print(forecast.fisher_numpy[:,:,4])
# print(forecast.getIntegratedFisher(forecast.K, forecast.fisher_numpy[0,0,:], 0.045,0.05, 1e9*values['survey_config']['geometry']['volume']))
# print(forecast.getIntegratedFisher(forecast.K, forecast.fisher_numpy[0,1,:], 0.045,0.05, 1e9*values['survey_config']['geometry']['volume']))
# print(forecast.getIntegratedFisher(forecast.K, forecast.fisher_numpy[1,1,:], 0.045,0.05, 1e9*values['survey_config']['geometry']['volume']))
# for i in range(len(variables_list_fisher)):
#     for j in range(i,len(variables_list_fisher)):
#         plt.plot(forecast.K,forecast.fisher_numpy[i,j,:],label='%d,%d' % (i,j))
# plt.xscale('log')
# plt.legend()
# plt.savefig('/Users/sforeman/Desktop/fish.pdf')

kf,sig_fnl = forecast.get_error('fnl', marginalized = False, integrated = False,
              kmin = K.min(), kmax = K.max(),
              volume = values['survey_config']['geometry']['volume'])
# np.savetxt(direc+data_dir+'sigma_fnl_unmarg_perk.dat',np.array((kf,sig_fnl*fnlScaling)).T)
np.savetxt(direc+data_dir+'sigma_fnl_unmarg_perk.dat',np.c_[kf,sig_fnl*fnlScaling])

kf,sig_fnl = forecast.get_error('fnl', marginalized = False, integrated = True,
              kmin = K.min(), kmax = K.max(),
              volume = values['survey_config']['geometry']['volume'],
              log_integral = True, mu_limit = mu_limit)
np.savetxt(direc+data_dir+'sigma_fnl_unmarg_int.dat',np.c_[kf,sig_fnl*fnlScaling])

# print(kf[0],sig_fnl[0])
# print(forecast.getIntegratedFisher(K, forecast.fisher_numpy[1,1, :], 0.001,0.1,
#                                 values['survey_config']['geometry']['volume']*1e9) )

kf,sig_fnl = forecast.get_error('fnl', marginalized = True, integrated = True,
              kmin = K.min(), kmax = K.max(),
              volume = values['survey_config']['geometry']['volume'],
              log_integral = True, mu_limit = mu_limit)
np.savetxt(direc+data_dir+'sigma_fnl_marg_int.dat',np.c_[kf,sig_fnl*fnlScaling])

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


b2_bs2_frac_prior_list = [0.2, 0.1, 0.05, 0.01]
if False:
    for frac_prior in b2_bs2_frac_prior_list:
        print('Recomputing marginalized forecasts with fractional b2,bs2 priors of %g' \
                % frac_prior)
        for v in ['b20','bs2']:
            forecast.inv_priors[v] = 1 / (var_values[v]*frac_prior)**2.

        kf,sig_fnl = forecast.get_error('fnl', marginalized = True, integrated = True,
                      recalculate=True, kmin = K.min(), kmax = K.max(),
                      volume = values['survey_config']['geometry']['volume'],
                      log_integral = True, mu_limit = mu_limit)
        np.savetxt(direc+data_dir+'sigma_fnl_marg_int_fracprior%g.dat' % frac_prior,
                    np.c_[kf,sig_fnl*fnlScaling])
