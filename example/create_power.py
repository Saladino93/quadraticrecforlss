import numpy as np

import camb
from camb import model, initialpower

import opentext

import os
import sys

from shutil import copyfile

def getpowerspectrum(z = [0.], minkh = 1e-4, maxkh = 10, nonlinear = True, npoints = 200):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.set_dark_energy() #omit?
    pars.InitPower.set_params(ns=0.965)
    pars.set_matter_power(redshifts = z, kmax = 2)
    if nonlinear:
    	pars.NonLinear = model.NonLinear_both
    else:
	pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh = minkh, maxkh = maxkh, npoints = npoints)
    return kh_nonlin, z_nonlin, pk_nonlin


filename = 'startingvalues.txt'
values = opentext.get_values(filename)

print('Values are, ', values)

direc = values['name']

if not os.path.exists(direc):
    os.makedirs(direc)
    direc = direc+'/'
    os.makedirs(direc+'data_dir')
    os.makedirs(direc+'pics')
else:
    print('Baseline name already exist!')
    sys.exit()

minkh = values['minkh']#1e-7
maxkh = values['maxkh']#1

#Redshift
z = np.array([values['z']])

npoints = int(values['npoints'])

print('Getting Theoretical Non Linear Power Spectrum')

power_arr_nl = getpowerspectrum(z = z, minkh = minkh, maxkh = maxkh, nonlinear = True, npoints = npoints)

th_nlK = power_arr_nl[0]
th_nlpower = power_arr_nl[2][0]

print('Done')

print('Getting Theoretical Linear Matter Power Spectrum')

power_arr = getpowerspectrum(z = z, minkh = minkh, maxkh = maxkh, nonlinear = False, npoints = npoints)

linear_power = power_arr[2][0]
K_linear = power_arr[0]

print('Done')

np.savetxt(direc+'linear_power.txt', np.c_[K_linear, linear_power])
np.savetxt(direc+'nonlinear_power.txt', np.c_[th_nlK, th_nlpower])
#np.savetxt(direc+'total_power.txt', np.c_[K, P_tot])
#np.savetxt(direc+'total_withlinear_power.txt', np.c_[K_linear, P_totlinear])

copyfile('startingvalues.txt', direc+'values.txt')
print('Finished')
