# Quadratic Reconstruction for LSS

Cosmology Project: Large Scale Structure of the the Primordial Universe

## Goal

Reconstructing the initial matter density field before non linear gravitational evolution. Use perturbation theory and reconstruction methods of fields. Then extract primordial local non Gaussianity parameter $f_{nl}$.

## Updates

02/06/2025
* Corrected small bug in tidal term. While this changes the specifics of effective biases and trispectrum shot noise calculations, main arguments are still true.

### Dependencies

Python3

numpy, scipy, sympy, matplotlib, nbodykit, mpmath (should be included with sympy).

### Installation

Use git clone then in the main directory do:

python setup.py install --user  (or without --user if you want)

### Usage

See examples directory. Just do: python run.py config_file_name.yaml
