import numpy as np

#nbodykit library for matter power spectra
from nbodykit.lab import *
import nbodykit

#system and os utilities
import os
import sys

#for reading config file
import yaml


''' START OF UTILITY FUNCTIONS '''

## This function helps you get the transfer function in function of redshift and at the desidred wave numbers specified by vector
def getT(cosmology, redshift, vector):
    transfer = nbodykit.cosmology.power.transfers.CLASS(cosmology, redshift = redshift)
    T = transfer.__call__(vector)
    return T

## This function helps you get the conversion factor M between the matter density contrast and the potential fluctuation
def getM(cosmology, redshift, vector):
    T = getT(cosmology, redshift, vector)
    Omega_m = cosmology.Omega0_m
    H0 = cosmology.H0
    lightvel = 3.*10**5.
    M = (2*lightvel**2./(3*H0**2.*Omega_m))*vector**2.*T
    return M

## Simple constant that appears in M
def get_alpha(cosmology):
    lightvel = 3.*10**5.
    H0 = cosmology.H0*cosmology.h
    Omega_m = cosmology.Omega0_m
    return (2*lightvel**2./(3*H0**2.*Omega_m))

''' END OF UTILITY FUNCTIONS '''



if len(sys.argv) == 1:
    print('Choose your configuration file!')
    sys.exit()

## Read configuration file

values_file = str(sys.argv[1])

with open(values_file, 'r') as stream:
    data = yaml.safe_load(stream)

values = data

print('Values are, ', values)

direc = values['name']
base_dir = values['file_config']['base_dir']
data_dir = values['file_config']['data_dir']
pics = values['file_config']['pics']

direc = base_dir+direc+'/'

if not os.path.exists(direc):
    os.makedirs(direc)
    os.makedirs(direc+data_dir)
    os.makedirs(direc+pics)
else:
    print('Baseline name already exists!')
    sys.exit()

#min and max wave numbers for definition of ranges  and the number of points in between to sample from
mink = float(values['data_creation_config']['mink'])
maxk = float(values['data_creation_config']['maxk'])
npoints = int(values['data_creation_config']['npoints'])

#Redshift
z = np.array([int(values['data_creation_config']['z'])])

#Cosmology Used
cosmology_used = values['data_creation_config']['cosmology']
if cosmology_used == 'Planck15':
    cosmo = cosmology.Planck15
elif cosmology_used == 'Planck13':
    cosmo = cosmology.Planck13


delta = (maxk-mink)/npoints
vector = np.arange(mink, maxk, delta)

#Obtain powers
print('Getting Power Spectra and Transfer Function')

vector_h = vector

powerhalo = nbodykit.cosmology.power.halofit.HalofitPower(cosmo, z)
powerhalo = powerhalo(vector_h)

powerlin = nbodykit.cosmology.power.LinearPower(cosmo, z)
powerlin = powerlin(vector_h)

M = getM(cosmo, redshift = z, vector = vector_h)

print('Done')

np.savetxt(direc+data_dir+values['file_config']['linear_power_name'], np.c_[vector, powerlin])
np.savetxt(direc+data_dir+values['file_config']['nonlinear_power_name'], np.c_[vector, powerhalo])
np.savetxt(direc+data_dir+values['file_config']['M_name'], np.c_[vector, M])

print('Finished')
