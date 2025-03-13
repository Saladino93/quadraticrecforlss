import numpy as np
import sympy as sp
import scipy.interpolate
import argparse
import os
import yaml
import sys
import pandas as pd
from quadratic_formula import QuadraticEstimator

def main():
    """
    Script to calculate quadratic estimator variances for multiple estimators
    and their cross-correlations using Vegas integration.
    
    Results are saved to a pandas DataFrame for easy analysis.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate quadratic estimator variances')
    parser.add_argument('config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--output', type=str, default='quadratic_results.pkl', 
                        help='Output file name for pandas DataFrame (default: quadratic_results.pkl)')
    args = parser.parse_args()
    
    # Read configuration file
    with open(args.config, 'r') as stream:
        values = yaml.safe_load(stream)
    print(f'Using config file: {args.config}')
    
    # Extract key parameters
    mink = values['analysis_config']['mink_reconstruction']
    maxk = values['analysis_config']['maxk_reconstruction']
    mink_analysis = values['analysis_config']['mink_analysis']
    maxk_analysis = values['analysis_config']['maxk_analysis']
    
    # Extract bias parameters
    b10 = float(values['survey_config']['tracer_properties']['biases']['b10'])
    b20 = float(values['survey_config']['tracer_properties']['biases']['b20'])
    bs2 = values['survey_config']['tracer_properties']['biases']['bs2']
    
    # If bs2 is not specified, use theory value
    if bs2 == '':
        print('Using theory value for bs2!')
        bs2 = -2./7. * (b10 - 1)
    else:
        bs2 = float(bs2)
    
    # Create an instance of the QuadraticEstimator
    estimator = QuadraticEstimator(mink, maxk)
    
    # Define symbols for F functions
    q1, q2, mu = sp.symbols('q1 q2 mu')
    
    # Define estimators and their coefficients
    estimator_configs = {
        'g': {
            'F': 17./21.,
            'ca': 1,#b10 + 21/17 * b20,
            'cb': 1#b10
        },
        's': {
            'F': 0.5*(q2/q1+q1/q2)*mu,
            'ca': 1,#b10,
            'cb': 1#b10
        },
        't': {
            'F': (2./7.)*mu**2.-1./3.,
            'ca': 1,#b10 + 7/2 * bs2,
            'cb': 1#b10
        },
        'x': {
            #'F': 1/(q1+q2)*mu, 
            'F': 0.5*(q2/q1+q1/q2)*mu,
            'ca': 1,#b10 + 7/2 * bs2,
            'cb': 0.001 #b10
        }
    }
    
    # Add F functions to the estimator
    for key, config in estimator_configs.items():
        print(f"Adding estimator '{key}'")
        estimator.addF(key, config['F'], ca=config['ca'], cb=config['cb'])
    
    # Load power spectra from files
    linear_power_file = os.path.join(
        values['file_config']['base_dir'], 
        values['name'], 
        values['file_config']['data_dir'], 
        values['file_config']['linear_power_name']
    )
    
    nonlinear_power_file = os.path.join(
        values['file_config']['base_dir'], 
        values['name'], 
        values['file_config']['data_dir'], 
        values['file_config']['nonlinear_power_name']
    )
    
    # Load linear power spectrum
    linear_data = np.loadtxt(linear_power_file)
    k_lin = linear_data[:, 0]
    pk_mm_linear = linear_data[:, 1]
    k_values = k_lin
    
    # Load nonlinear power spectrum
    nonlinear_data = np.loadtxt(nonlinear_power_file)
    k_nl = nonlinear_data[:, 0]
    pk_mm_nonlinear = nonlinear_data[:, 1]
    
    # Interpolate to our k_values if needed
    if len(k_nl) != len(k_values) or not np.allclose(k_nl, k_values):
        pk_mm_nonlinear = np.interp(k_values, k_nl, pk_mm_nonlinear)
    
    # Get shot noise from config
    nhalo = float(values['survey_config']['tracer_properties']['nhalo'])
    shot_noise = 1.0 / nhalo
    
    # Create galaxy power spectra
    pk_galgal_nonlinear = b10**2 * pk_mm_nonlinear
    pk_galgal_total = pk_galgal_nonlinear + shot_noise
    pk_mm_total = pk_mm_nonlinear
    pk_galm_nonlinear = b10 * pk_mm_nonlinear
    
    # Add power spectra to the estimator
    # Signal power spectra (linear)
    estimator.add_power_spectrum('mm', k_values, pk_mm_linear, 'signal')
    estimator.add_power_spectrum('galgal', k_values, pk_mm_linear, 'signal')
    estimator.add_power_spectrum('galm', k_values, pk_mm_linear, 'signal')
    estimator.add_power_spectrum('mgal', k_values, pk_mm_linear, 'signal')
    
    # Total power spectra (nonlinear + noise)
    estimator.add_power_spectrum('mm', k_values, pk_mm_total, 'total')
    estimator.add_power_spectrum('galgal', k_values, pk_galgal_total, 'total')
    estimator.add_power_spectrum('galm', k_values, pk_galm_nonlinear, 'total')
    estimator.add_power_spectrum('mgal', k_values, pk_galm_nonlinear, 'total')
    
    # Define k values of interest
    num_k_points = values['analysis_config'].get('num_k_points', 30)
    K_values = np.geomspace(mink_analysis, maxk_analysis, num_k_points)
    
    # Store K values in estimator.Krange for consistency with asymmgen.py
    estimator.Krange = K_values
    
    # Vegas integration parameters
    nitn = values['analysis_config'].get('nitn', 60)
    neval = values['analysis_config'].get('neval', 500)
    
    # Define tracer combinations
    tracer_combinations = [('gal', 'gal')]  # Changed from ('g', 'g') to ('gal', 'gal')
    
    # Create a list of all estimator pairs to calculate
    estimator_keys = list(estimator_configs.keys())
    estimator_pairs = []
    for i, alpha in enumerate(estimator_keys):
        for beta in estimator_keys[i:]:  # Only compute upper triangle (including diagonal)
            estimator_pairs.append((alpha, beta))
    
    print(f"Calculating {len(estimator_pairs)} estimator pairs: {estimator_pairs}")
    
    # Initialize results dictionary
    results = {
        'K': K_values,
    }
    
    # Calculate variance estimators for all pairs
    for alpha, beta in estimator_pairs:
        for tracers in tracer_combinations:
            alpha_tracers = tracers
            beta_tracers = tracers
            
            key = f"{alpha}{beta}_{tracers[0]}{tracers[1]}"
            print(f"Computing variance for {key}...")
            
            means, errors = estimator.variance_estimators(
                alpha, beta, K_values, mink, maxk,
                alpha_tracers=alpha_tracers, beta_tracers=beta_tracers,
                nitn=nitn, neval=neval, show_progress=True
            )
            
            # Store only the means with a simplified key (no "_mean" suffix)
            results[key] = means
    
    # Convert results to pandas DataFrame
    df = pd.DataFrame(results)
    
    # Add tracer information as metadata
    df.attrs['tracer_combinations'] = tracer_combinations
    
    # Create output directory
    output_dir = os.path.join(values['file_config']['base_dir'], values['name'], values['file_config']['data_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to file in the appropriate directory
    output_file = os.path.join(output_dir, args.output)
    df.to_pickle(output_file)
    print(f"Results saved to {output_file}")
    
    # Also save individual results in text format for compatibility
    for alpha, beta in estimator_pairs:
        for tracers in tracer_combinations:
            key = f"{alpha}{beta}_{tracers[0]}{tracers[1]}"
            filename = os.path.join(output_dir, f"N{alpha}{beta}_{tracers[0]}{tracers[1]}.txt")
            means = results[key]
            np.savetxt(filename, np.array((K_values, means)).T)

if __name__ == "__main__":
    main() 