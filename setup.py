from distutils.core import setup, Extension
import os

setup(name='quadforlss',
      version='0.1',
      description='Cosmology Analysis',
      url='https://github.com/Saladino93/',
      author='Omar Darwish',
      author_email='od261@cam.ac.uk',
      license='BSD-2-Clause',
      packages=['quadforlss'],
      package_dir={'quadforlss':'estimator'},
      zip_safe=False)
