import numpy as np

import camb
from camb import model, initialpower

import opentext

import os
import sys

from shutil import copyfile

import nbodykit.lab as lab
import nbodykit


#######

c = lab.cosmology.Planck15
h = c.h
H0 = 100.*h
ombh2 = c.ombh2
omch2 = c.omch2
ns = c.ns


#######

vector = np.arange(0, 0.5, 0.0001)

def getT(cosmology, redshift, vector):
    transfer = nbodykit.cosmology.power.transfers.CLASS(cosmology, redshift = redshift)
    T = transfer.__call__(vector)
    return T


def getM(cosmology, redshift, vector):
    #T = getT(c, redshift, vector)
    Omega_m = cosmology.Omega0_m
    H0 = cosmology.H0
    lightvel = 3.*10**5.
    M = (2*lightvel**2./(3*H0**2.*Omega_m))*vector**2.*T
    return M
    
M = getM(c, redshift = 0., vector = vector)

Plin = cosmology.LinearPower(c, redshift = 0., transfer='EisensteinHu')
Kslin = np.arange(0.005, 0.5, 0.01)
Plin = Plin(Kslin)

np.savetxt('temp.txt',  np.c_[vector, M])
#######

#a lot copied from camb demo
def getpowerspectrum(z = [0.], minkh = 1e-4, maxkh = 10, nonlinear = True, npoints = 200):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0 = H0, ombh2 = ombh2, omch2 = omch2)
    pars.set_dark_energy() #omit?
    pars.InitPower.set_params(ns = ns)
    pars.set_matter_power(redshifts = z, kmax = 10.)
    if nonlinear:
        pars.NonLinear = model.NonLinear_both
    else:
        pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh = minkh, maxkh = maxkh, npoints = npoints)
    trans = results.get_matter_transfer_data()
    kh = trans.transfer_data[0, :, 0]
    delta = trans.transfer_data[model.Transfer_cdm-1, :, 0]
    #print(kh/kh_nonlin) 
    np.savetxt(direc+'linear_power.txt', np.c_[kh, delta])
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

np.savetxt(direc+'linear_power.txt', np.c_[K_linear*h, linear_power*h**-3.])
np.savetxt(direc+'nonlinear_power.txt', np.c_[th_nlK*h, th_nlpower*h**-3.])
#np.savetxt(direc+'total_power.txt', np.c_[K, P_tot])
#np.savetxt(direc+'total_withlinear_power.txt', np.c_[K_linear, P_totlinear])

copyfile('startingvalues.txt', direc+'values.txt')
print('Finished')
