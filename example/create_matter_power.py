import numpy as np

from nbodykit.lab import *
import nbodykit

import opentext

import os
import sys

from shutil import copyfile

def getT(cosmology, redshift, vector):
    transfer = nbodykit.cosmology.power.transfers.CLASS(cosmology, redshift = redshift)
    T = transfer.__call__(vector)
    return T

def getM(cosmology, redshift, vector):
    T = getT(cosmology, redshift, vector)
    Omega_m = cosmology.Omega0_m
    H0 = cosmology.H0 #multiply by c.h for units if vector is 1/Mpc
    lightvel = 3.*10**5.
    M = (2*lightvel**2./(3*H0**2.*Omega_m))*vector**2.*T
    return M 

def get_alpha(cosmology):
    lightvel = 3.*10**5.
    H0 = cosmology.H0*cosmology.h
    Omega_m = cosmology.Omega0_m
    return (2*lightvel**2./(3*H0**2.*Omega_m))

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

#min and max wave numbers for definition of ranges  and the number of points in between to sample from
mink = values['mink']
maxk = values['maxk']
npoints = values['npoints'] 

#Redshift
z = np.array([values['z']])

#Cosmology Used
cosmo = cosmology.Planck15

delta = (maxk-mink)/npoints
vector = np.arange(mink, maxk, delta) #This is in units of 1/Mpc

#Obtain powers
print('Getting Power Spectra and Transfer Function')

vector_h = vector*cosmo.h**-1. #wave number in units of h/Mpc

powerhalo = nbodykit.cosmology.power.halofit.HalofitPower(cosmo, z) 
powerhalo = powerhalo(vector_h) #this is in units of h^-3Mpc^3
powerhalo *= cosmo.h**-3. #to get units Mpc^3

powerlin = nbodykit.cosmology.power.LinearPower(cosmo, z)
powerlin = powerlin(vector_h) #this is in units of h^-3Mpc^3
powerlin *= cosmo.h**-3. #to get units Mpc^3

M = getM(cosmo, redshift = z, vector = vector_h) #this is in dimensionless units

print('Alpha is (Mpc**2), ', get_alpha(cosmo))

print('Done')

np.savetxt(direc+'linear_power.txt', np.c_[vector, powerlin])
np.savetxt(direc+'nonlinear_power.txt', np.c_[vector, powerhalo])
np.savetxt(direc+'M.txt', np.c_[vector, M])
copyfile('startingvalues.txt', direc+'values.txt')
print('Finished')
