import numpy as np
import sympy as sp
import functools
import scipy.interpolate
import vegas
from tqdm import tqdm

class vectorize(np.vectorize):
    """Vectorizes a function.
    """
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

class QuadraticEstimator:
    """
    A class implementing the quadratic estimator variance formula.
    
    This class provides a minimal implementation of the quadratic formula with
    weights w that depend on self.f and the total power spectra of tracers.
    """
    
    def __init__(self, minkhrec, maxkhrec):
        """
        Initialize the QuadraticEstimator class.
        """
        self.q1, self.q2, self.mu = sp.symbols('q1 q2 mu')
        self._F = {}
        self._Ffunc = {}
        self._coefficients = {}
        self.keys = []
        self.params = {}  # Dictionary to store parameters
        
        # Power spectrum dictionaries
        self.total_power = {}  # Total power spectra (signal + noise)
        self.signal_power = {}  # Signal power spectra
        
        # Cache for power spectrum values
        self._power_cache = {}

        self.minkhrec = minkhrec
        self.maxkhrec = maxkhrec
        self.bounds = lambda x: np.where(x<self.maxkhrec, np.where(x>self.minkhrec, 1., 0.), 0.)
        
    def add_power_spectrum(self, key, k_values, pk_values, power_type='total'):
        """
        Add a power spectrum to the appropriate dictionary and create an interpolation function.
        
        Parameters
        ----------
        key : str
            Key for the power spectrum (e.g., 'mm' for matter-matter).
        k_values : array-like
            Array of k values.
        pk_values : array-like
            Array of P(k) values corresponding to k_values.
        power_type : str, optional
            Type of power spectrum: 'total' or 'signal'.
        """

        index_max = np.where(k_values<self.maxkhrec)[0][-1]
        index_min = np.where(k_values>self.minkhrec)[0][0]

        k_values = k_values[index_min:index_max]
        pk_values = pk_values[index_min:index_max]

        # Create interpolation function
        log_interp = scipy.interpolate.interp1d(
            np.log(k_values), np.log(pk_values),
            fill_value = 0. if power_type == 'signal' else 1e2, bounds_error = False)
        
        
        # Define the interpolation function that will be used
        def pk_function(k_array):
            # Apply interpolation in log-log space
            log_pk = log_interp(np.log(k_array))
            pk = np.exp(log_pk)
            return pk

        """
        finterp = scipy.interpolate.interp1d(k_values, pk_values, fill_value = 0. if power_type == 'signal' else 1e10, bounds_error = False)
        def pk_function(k_array):
            pk = finterp(k_array)
            return pk
        """
        
        # Store the function in the appropriate dictionary
        if power_type.lower() == 'total':
            self.total_power[key] = pk_function
        elif power_type.lower() == 'signal':
            self.signal_power[key] = pk_function

        else:
            raise ValueError(f"Unknown power_type: {power_type}. Use 'total' or 'signal'.")
        
        # Clear the cache when adding a new power spectrum
        self._power_cache = {}
    
    def add_power_spectra_from_dict(self, power_dict, power_type='total'):
        """
        Add multiple power spectra from a dictionary.
        
        Parameters
        ----------
        power_dict : dict
            Dictionary where keys are spectrum names and values are dictionaries
            with 'k' and 'pk' keys containing the k values and P(k) values.
        power_type : str, optional
            Type of power spectrum: 'total' or 'signal'.
        """
        for key, data in power_dict.items():
            if 'k' in data and 'pk' in data:
                self.add_power_spectrum(key, data['k'], data['pk'], power_type)
            else:
                raise ValueError(f"Power spectrum data for {key} must contain 'k' and 'pk' keys.")
        
    def addF(self, key, F, ca=1.0, cb=1.0):
        """
        Add mode-coupling 'F' function to quadratic estimator.
        
        Parameters
        ----------
        key : str
            Key for mode-coupling (e.g. 't').
        F : sympy expression
            Algebraic form of mode-coupling F function.
        ca : float, optional
            Coefficient for the first term.
        cb : float, optional
            Coefficient for the second term.
        """
        self._F[key] = F
        self._Ffunc[key] = sp.lambdify([self.q1, self.q2, self.mu], self._F[key], 'numpy')
        self._coefficients[key] = (ca, cb)
        self.keys = list(self._F.keys())
        
    def getF(self, key):
        """
        Get stored expression for a particular mode-coupling function.
        
        Parameters
        ----------
        key : str
            Key for desired expression (e.g. 't').
            
        Returns
        -------
        exp : sympy expression
            Algebraic form of F function.
        """
        return self._F[key]
    
    def getF_func(self, key):
        """
        Get function for a particular mode-coupling function.
        
        Parameters
        ----------
        key : str
            Key for desired function.
            
        Returns
        -------
        func : function
            Function form of F function.
        """
        return self._Ffunc[key]
    
    def set_params(self, **kwargs):
        """
        Set parameters for the model.
        
        Parameters
        ----------
        **kwargs : dict
            Dictionary of parameter names and values.
        """
        for key, value in kwargs.items():
            self.params[key] = value
    
    def get_param(self, key, default=None):
        """
        Get a parameter value.
        
        Parameters
        ----------
        key : str
            Parameter name.
        default : any, optional
            Default value if parameter is not found.
            
        Returns
        -------
        value : any
            Parameter value.
        """
        return self.params.get(key, default)
    
    def get_power(self, k, tracer_key='mm', power_type='total'):
        """
        Get power spectrum value for a given k and tracer combination.
        
        This method is memoized for both scalar and array k values to improve performance.
        
        Parameters
        ----------
        k : float or array-like
            Wavenumber(s).
        tracer_key : str, optional
            Tracer combination key (e.g., 'mm' for matter-matter, 'hm' for halo-matter).
        power_type : str, optional
            Type of power spectrum: 'total' or 'signal'.
            
        Returns
        -------
        P : float or array-like
            Power spectrum value(s).
        """
        if power_type.lower() == 'total':
            if tracer_key in self.total_power:
                return self.total_power[tracer_key](k)
            else:
                raise KeyError(f"Total power spectrum for tracers '{tracer_key}' not found.")
        elif power_type.lower() == 'signal':
            if tracer_key in self.signal_power:
                return self.signal_power[tracer_key](k)
            else:
                raise KeyError(f"Signal power spectrum for tracers '{tracer_key}' not found.")
        else:
            raise ValueError(f"Unknown power_type: {power_type}. Use 'total' or 'signal'.")
    
    def clear_cache(self):
        """
        Clear the power spectrum cache.
        
        This can be useful when memory usage becomes a concern or when
        power spectra are updated.
        """
        # Clear the lru_cache for the get_power method
        self.get_power.cache_clear()
        
        # For backward compatibility, also clear the old cache dictionary
        self._power_cache = {}
    
    def f_vectors(self, kernel_key, k1, k2, tracer_A='m', tracer_B='m'):
        """
        Compute the mode-coupling 'f' function for general k1 and k2 vectors.
        
        This implementation is optimized for batch processing with Vegas integration.
        It assumes k1 and k2 are 3D vectors, potentially with batch dimensions.
        
        Parameters
        ----------
        kernel_key : str
            Key for F function of interest.
        k1 : array-like
            First wavevector (3D vector or batch of 3D vectors).
        k2 : array-like
            Second wavevector (3D vector or batch of 3D vectors).
        tracer_A : str, optional
            First tracer type (e.g., 'm' for matter, 'h' for halo).
        tracer_B : str, optional
            Second tracer type (e.g., 'm' for matter, 'h' for halo).
            
        Returns
        -------
        result : float or array
            Value of the f function.
        prod_alpha_ab : float or array
            First term of the f function.
        prod_alpha_ba : float or array
            Second term of the f function.
        """
        # Get the F kernel function and coefficients
        Fkernel = self._Ffunc[kernel_key]
        c1, c2 = self._coefficients[kernel_key]
        
        # Calculate k3 = k1 + k2 (the total wavevector)
        k3 = k1 + k2 #K
        
        # Calculate magnitudes
        k1_mag = np.linalg.norm(k1, axis=-1)
        k2_mag = np.linalg.norm(k2, axis=-1)
        k3_mag = np.linalg.norm(k3, axis=-1)
        
        # Calculate cosines of angles between vectors
        # For F(k3, k1, mu_31), mu_31 is the cosine between k3 and -k1
        # We need the negative of k1 for the kernel
        mu_31 = -np.sum(k3 * k1, axis=-1) / (k3_mag * k1_mag)
        
        # For F(k3, k2, mu_32), mu_32 is the cosine between k3 and -k2
        # We need the negative of k2 for the kernel
        mu_32 = -np.sum(k3 * k2, axis=-1) / (k3_mag * k2_mag)
        
        # Get the signal power spectra for the tracers
        P_AB = self.get_power(k1_mag, f"{tracer_A}{tracer_B}", 'signal')
        P_BA = self.get_power(k2_mag, f"{tracer_B}{tracer_A}", 'signal')
        
        # Calculate the f function terms with power spectra
        # Following the pattern in estimator.py's f_general method
        prod_alpha_ab = Fkernel(k3_mag, k1_mag, mu_31) * P_AB
        prod_alpha_ba = Fkernel(k3_mag, k2_mag, mu_32) * P_BA
        
        # Sum the terms and multiply by 2
        result = (c1 * prod_alpha_ab + c2 * prod_alpha_ba) * 2
        
        return result, prod_alpha_ab, prod_alpha_ba
    
    def f(self, kernel_key, q, K, mu, tracer_A='m', tracer_B='m'):
        """
        Compute the mode-coupling 'f' function.
        
        This follows the pattern in estimator.py, where the f function includes
        the signal power spectra in its calculation.
        
        Parameters
        ----------
        kernel_key : str
            Key for F function of interest.
        q : float or array
            Vector norm (k1).
        K : float or array
            Vector norm (k3).
        mu : float or array
            Angle between vectors.
        tracer_A : str, optional
            First tracer type (e.g., 'm' for matter, 'h' for halo).
        tracer_B : str, optional
            Second tracer type (e.g., 'm' for matter, 'h' for halo).
            
        Returns
        -------
        result : float or array
            Value of the f function.
        prod_alpha_ab : float or array
            First term of the f function.
        prod_alpha_ba : float or array
            Second term of the f function.
        """
        # Calculate the magnitude of k_2 = |K-q|
        modK_q = np.sqrt(K**2 + q**2 - 2*K*q*mu)
        
        # Calculate the cosine of the angle between k_3 and -k_2
        mu_32 = -(K**2 - q*K*mu) / (K*modK_q)
        
        # Get the F kernel function and coefficients
        Fkernel = self._Ffunc[kernel_key]
        c1, c2 = self._coefficients[kernel_key]
        
        # Get the signal power spectra for the tracers
        P_AB = self.get_power(q, f"{tracer_A}{tracer_B}", 'signal')
        P_BA = self.get_power(modK_q, f"{tracer_B}{tracer_A}", 'signal')
        
        # Calculate the f function terms with power spectra
        # Following the pattern in estimator.py's f_general method
        prod_alpha_ab = Fkernel(K, q, -mu) * P_AB #F(K, -q)
        prod_alpha_ba = Fkernel(K, modK_q, mu_32) * P_BA #F(K, q-K)
        
        # Sum the terms and multiply by 2
        result = (c1 * prod_alpha_ab + c2 * prod_alpha_ba) * 2
        
        return result, prod_alpha_ab, prod_alpha_ba
    
    def w_vectors(self, kernel_key, k1, k2, tracer_A='m', tracer_B='m', power_type='total'):
        """
        Weight function w that depends on self.f and the total power spectra of tracers.
        
        The weight is given by: f/(2*PAA_tot(k1)*PBB_tot(k2))
        
        This implementation is optimized for batch processing with Vegas integration.
        It assumes k1 and k2 are 3D vectors, potentially with batch dimensions.
        
        Parameters
        ----------
        kernel_key : str
            Key for F function of interest.
        k1 : array-like
            First wavevector (3D vector or batch of 3D vectors).
        k2 : array-like
            Second wavevector (3D vector or batch of 3D vectors).
        tracer_A : str, optional
            First tracer type (e.g., 'm' for matter, 'h' for halo).
        tracer_B : str, optional
            Second tracer type (e.g., 'm' for matter, 'h' for halo).
        power_type : str, optional
            Type of power spectrum to use: 'total' or 'signal'.
            
        Returns
        -------
        result : float or array
            Value of the weight function.
        """
        # Calculate magnitudes
        k1_mag = np.linalg.norm(k1, axis=-1)
        k2_mag = np.linalg.norm(k2, axis=-1)
                
        # Get the f function value using vectors
        f_value, _, _ = self.f_vectors(kernel_key, k1, k2, tracer_A, tracer_B)
        
        # Get power spectrum values for the tracers
        P_AA = self.get_power(k1_mag, f"{tracer_A}{tracer_A}", power_type)
        P_BB = self.get_power(k2_mag, f"{tracer_B}{tracer_B}", power_type)
        
        # Calculate the weight function according to the formula
        # w_α(k1,k2) = f_α(k1,k2) / (2*PAA_tot(k1)*PBB_tot(k2))
        denominator = 2.0 * P_AA * P_BB
        
        result = f_value / denominator
        
        return result
    
    def w(self, kernel_key, q, K, mu, tracer_A='m', tracer_B='m', power_type='total'):
        """
        Weight function w that depends on self.f and the total power spectra of tracers.
        
        The weight is given by: f/(2*PAA_tot(k1)*PBB_tot(k2))
        
        Parameters
        ----------
        kernel_key : str
            Key for F function of interest.
        q : float or array
            Vector norm (k1).
        K : float or array
            Vector norm (k3).
        mu : float or array
            Angle between vectors.
        tracer_A : str, optional
            First tracer type (e.g., 'm' for matter, 'h' for halo).
        tracer_B : str, optional
            Second tracer type (e.g., 'm' for matter, 'h' for halo).
        power_type : str, optional
            Type of power spectrum to use: 'total' or 'signal'.
            
        Returns
        -------
        result : float or array
            Value of the weight function.
        """
        # Get the f function value (first return value)
        f_value, _, _ = self.f(kernel_key, q, K, mu, tracer_A, tracer_B)
        
        # Calculate the magnitude of k_2 = |k_1 + k_3 - k_1|
        modK_q = np.sqrt(K**2 + q**2 - 2*K*q*mu)
        
        # Get power spectrum values for the tracers
        # We need PAA_tot(k1) and PBB_tot(k2)
        P_AA = self.get_power(q, f"{tracer_A}{tracer_A}", power_type)
        P_BB = self.get_power(modK_q, f"{tracer_B}{tracer_B}", power_type)
        
        # Calculate the weight function according to the formula
        # w_α(k1,k2) = f_α(k1,k2) / (2*PAA_tot(k1)*PBB_tot(k2))
        denominator = 2.0 * P_AA * P_BB
        
        result = f_value / denominator
        
        return result
    
    def integrand(self, q, mu, K, alpha, beta, 
                 alpha_tracers=('m', 'm'), beta_tracers=('m', 'm'),
                 power_type='total'):
        """
        Define the integrand expression for the quadratic formula.
        
        The integrand is:
        w_{alpha}^{AB}(k1, k2)*[w_{beta}^{XY}(k1, k2)P_{AX}(k1)P_{BY}(k2)+w_{beta}^{XY}(k2, k1)P_{AY}(k1)P_{BX}(k2)]
        
        Where:
        - w_{alpha}^{AB} and w_{beta}^{XY} are the weight functions
        - k1 is the integration variable (q)
        - k2 is calculated from k1, k3, and μ
        - k3 is the external wavenumber (K)
        - P_{AX}, P_{BY}, etc. are power spectra between different tracers
        
        Parameters
        ----------
        q : float
            Wavenumber k1 (integration variable).
        mu : float
            Cosine of the angle between k1 and k3.
        K : float
            External wavenumber k3.
        alpha : str
            Key for first F function.
        beta : str
            Key for second F function.
        alpha_tracers : tuple, optional
            Tracers for alpha weight (tracer_A, tracer_B).
        beta_tracers : tuple, optional
            Tracers for beta weight (tracer_X, tracer_Y).
        power_type : str, optional
            Type of power spectrum to use: 'total' or 'signal'.
            
        Returns
        -------
        result : float
            Value of the integrand.
        """
        # Extract tracers
        tracer_A, tracer_B = alpha_tracers
        tracer_X, tracer_Y = beta_tracers
        
        # Calculate k2 (modK_q)
        modK_q = np.sqrt(K**2 + q**2 - 2*K*q*mu)
        
        # Calculate mu' for the reversed case (k2, k1)
        mu_p = K**2 - q*K*mu
        mu_p /= (K*modK_q)

        # Calculate the weight functions
        w_alpha_AB = self.w(
            alpha, q, K, mu, 
            tracer_A=tracer_A, tracer_B=tracer_B,
            power_type=power_type
        )
        
        # First term: w_{beta}^{XY}(k1, k2)
        w_beta_XY_k1_k2 = self.w(
            beta, q, K, mu, 
            tracer_A=tracer_X, tracer_B=tracer_Y,
            power_type=power_type
        )
        
        # Second term: w_{beta}^{XY}(k2, k1)
        # Note: For the reversed case, we need to swap q and modK_q, and use mu_p
        w_beta_XY_k2_k1 = self.w(
            beta, modK_q, K, -mu_p,  # Negative mu_p because of the direction change
            tracer_A=tracer_X, tracer_B=tracer_Y,
            power_type=power_type
        )
        
        # Get power spectrum values
        P_AX = self.get_power(q, f"{tracer_A}{tracer_X}", power_type)
        P_BY = self.get_power(modK_q, f"{tracer_B}{tracer_Y}", power_type)
        P_AY = self.get_power(q, f"{tracer_A}{tracer_Y}", power_type)
        P_BX = self.get_power(modK_q, f"{tracer_B}{tracer_X}", power_type)

        #w_alpha_AB_ = self.f(alpha, q, K, mu, tracer_A, tracer_B)[0]**2/(2*P_AX*P_BY)
        #w_alpha_AB = resultf**2/(2*P_AX*P_BY)
        #         
        # Calculate the terms in the brackets
        bounds = self.bounds(q) * self.bounds(modK_q)
        term1 = P_AX * P_BY * w_beta_XY_k1_k2
        term2 = P_AY * P_BX * w_beta_XY_k1_k2 #w_beta_XY_k2_k1
        
        factor = 2*np.pi*q**2./(2*np.pi)**3.
        # Calculate the full integrand
        result = factor * w_alpha_AB * bounds * (term1 + term2)
        
        return result
    

    def integrand_projection(self, q, mu, K, alpha, beta, 
                 alpha_tracers=('m', 'm'),
                 power_type='total'):
        """
        It calculates \int_{k} weight_alpha * f_beta.
        """

        w_alpha_AB = self.w(alpha, q, K, mu, alpha_tracers[0], alpha_tracers[1], power_type)
        f_beta_XY, _, _ = self.f(beta, q, K, mu, alpha_tracers[0], alpha_tracers[1])
        
        return w_alpha_AB * f_beta_XY
        
    
    def integrand_vec(self, q, mu, K, alpha, beta, 
                 alpha_tracers=('m', 'm'), beta_tracers=('m', 'm'),
                 power_type='total'):
        """
        This calculates a general variance, defined as:
        
        w_{alpha}^{AB}(k1, k2)*[w_{beta}^{XY}(k1, k2)P_{AX}(k1)P_{BY}(k2)+w_{beta}^{XY}(k2, k1)P_{AY}(k1)P_{BX}(k2)]
        
        This implementation is optimized for batch processing with Vegas integration.
        It converts q and mu values to 3D vectors and uses the vector-based weight function.
        
        In the quadratic estimator formalism:
        - k1 is the q vector (integration variable)
        - k3 is the K vector (external wavenumber)
        - k2 = k3 - k1 (the difference vector)
        
        
        Parameters
        ----------
        q : array-like
            Wavenumber k1 (integration variable), can be a batch of values.
        mu : array-like
            Cosine of the angle between k1 and k3, can be a batch of values.
        K : float
            External wavenumber k3 (scalar value).
        alpha : str
            Key for first F function.
        beta : str
            Key for second F function.
        alpha_tracers : tuple, optional
            Tracers for alpha weight (tracer_A, tracer_B).
        beta_tracers : tuple, optional
            Tracers for beta weight (tracer_X, tracer_Y).
        power_type : str, optional
            Type of power spectrum to use: 'total' or 'signal'.
            
        Returns
        -------
        result : array-like
            Value of the integrand for each point in the batch.
        """
        # Extract tracers
        tracer_A, tracer_B = alpha_tracers
        tracer_X, tracer_Y = beta_tracers
        
        # Ensure q and mu are arrays for batch processing
        q = np.atleast_1d(q)
        mu = np.atleast_1d(mu)
        
        # Set up coordinate system:
        # - Place k3 (K vector) along the z-axis: k3 = [0, 0, K]
        # - Place k1 (q vector) in the x-z plane with angle mu to k3
        
        # Calculate the x and z components of k1 (q vector)
        # k1 = q * [sin(theta), 0, cos(theta)] where cos(theta) = mu
        k1_x = q * np.sqrt(1 - mu**2)  # sin(theta) = sqrt(1 - mu^2)
        k1_z = q * mu
        
        # Create the k1 vector [k1_x, 0, k1_z]
        k1 = np.stack([k1_x, np.zeros_like(q), k1_z], axis=-1)
        
        # Create the k3 vector [0, 0, K]
        k3 = np.zeros_like(k1)
        k3[..., 2] = K  # Set the z-component to K
        
        # Calculate k2 = k3 - k1 (the difference vector)
        k2 = k3 - k1
        
        # Calculate magnitudes
        k1_mag = q  # We already know this is q
        k2_mag = np.linalg.norm(k2, axis=-1)  # |K-q|
        
        # Calculate the weight functions using the vector-based methods
        w_alpha_AB = self.w_vectors(
            alpha, k1, k2, 
            tracer_A=tracer_A, tracer_B=tracer_B,
            power_type=power_type
        )
        
        # For the second term, we need to swap k1 and k2
        w_beta_XY_k1_k2 = self.w_vectors(
            beta, k1, k2, 
            tracer_A=tracer_X, tracer_B=tracer_Y,
            power_type=power_type
        )
        
        w_beta_XY_k2_k1 = self.w_vectors(
            beta, k2, k1,
            tracer_A=tracer_X, tracer_B=tracer_Y,
            power_type=power_type
        )
        
        # Get power spectrum values
        P_AX = self.get_power(k1_mag, f"{tracer_A}{tracer_X}", power_type)
        P_BY = self.get_power(k2_mag, f"{tracer_B}{tracer_Y}", power_type)
        P_AY = self.get_power(k1_mag, f"{tracer_A}{tracer_Y}", power_type)
        P_BX = self.get_power(k2_mag, f"{tracer_B}{tracer_X}", power_type)
        
        # Apply bounds to ensure we're within the valid k range
        bounds = self.bounds(k1_mag) * self.bounds(k2_mag)
        
        # Calculate the terms in the brackets
        term1 = P_AX * P_BY * w_beta_XY_k1_k2
        term2 = P_AY * P_BX * w_beta_XY_k2_k1

        #print(w_beta_XY_k1_k2/w_beta_XY_k2_k1)
        # Calculate the full integrand
        # The factor includes the Jacobian for spherical integration
        factor = 2*np.pi*q**2./(2*np.pi)**3.
        result = factor * w_alpha_AB * bounds * (term1 + term2)
        
        return result
    
    def integrand_projection_vec(self, q, mu, K, alpha, beta,
                 alpha_tracers=('m', 'm'),
                 power_type='total'):
        """
        It calculates \int_{k} weight_alpha * f_beta.
        
        This implementation is optimized for batch processing with Vegas integration.
        It converts q and mu values to 3D vectors and uses the vector-based weight function.
        
        In the quadratic estimator formalism:
        - k1 is the q vector (integration variable)
        - k3 is the K vector (external wavenumber)
        - k2 = k3 - k1 (the difference vector)
        
        The integrand is:
        \int_{k1} w_{alpha}^{AB}(k1, k2)f_beta(k1, k2)
        
        Parameters
        ----------
        q : array-like
            Wavenumber k1 (integration variable), can be a batch of values.
        mu : array-like
            Cosine of the angle between k1 and k3, can be a batch of values.
        K : float
            External wavenumber k3 (scalar value).
        alpha : str
            Key for first F function.
        beta : str
            Key for second F function.
        alpha_tracers : tuple, optional
            Tracers for alpha weight (tracer_A, tracer_B).
        beta_tracers : tuple, optional
            Tracers for beta weight (tracer_X, tracer_Y).
        power_type : str, optional
            Type of power spectrum to use: 'total' or 'signal'.
            
        Returns
        -------
        result : array-like
            Value of the integrand for each point in the batch.
        """
        # Extract tracers
        tracer_A, tracer_B = alpha_tracers

        # Ensure q and mu are arrays for batch processing
        q = np.atleast_1d(q)
        mu = np.atleast_1d(mu)
        
        # Set up coordinate system:
        # - Place k3 (K vector) along the z-axis: k3 = [0, 0, K]
        # - Place k1 (q vector) in the x-z plane with angle mu to k3
        
        # Calculate the x and z components of k1 (q vector)
        # k1 = q * [sin(theta), 0, cos(theta)] where cos(theta) = mu
        k1_x = q * np.sqrt(1 - mu**2)  # sin(theta) = sqrt(1 - mu^2)
        k1_z = q * mu
        
        # Create the k1 vector [k1_x, 0, k1_z]
        k1 = np.stack([k1_x, np.zeros_like(q), k1_z], axis=-1)
        
        # Create the k3 vector [0, 0, K]
        k3 = np.zeros_like(k1)
        k3[..., 2] = K  # Set the z-component to K
        
        # Calculate k2 = k3 - k1 (the difference vector)
        k2 = k3 - k1
        
        # Calculate magnitudes
        k1_mag = q  # We already know this is q
        k2_mag = np.linalg.norm(k2, axis=-1)  # |K-q|
        
        # Calculate the weight functions using the vector-based methods
        w_alpha_AB = self.w_vectors(
            alpha, k1, k2, 
            tracer_A=tracer_A, tracer_B=tracer_B,
            power_type=power_type
        )

        f_beta_AB, _, _ = self.f_vectors(beta, k1, k2, tracer_A, tracer_B)
        
        # Apply bounds to ensure we're within the valid k range
        bounds = self.bounds(k1_mag) * self.bounds(k2_mag)
        
        factor = 2*np.pi*q**2./(2*np.pi)**3.
        result = factor * w_alpha_AB * f_beta_AB * bounds
        return result
    

    
    def _outer_integral_vegas(self, K, alpha, beta, 
                             alpha_tracers=('m', 'm'), beta_tracers=('m', 'm'),
                             power_type='total', case = "variance"):
        """
        Create a Vegas batch integrand function for the quadratic formula.
        
        This follows the pattern in estimator.py where q and mu are defined inside
        the integrand function rather than passed as parameters.
        
        Parameters
        ----------
        K : float
            External wavenumber k3.
        alpha : str
            Key for first F function.
        beta : str
            Key for second F function.
        alpha_tracers : tuple, optional
            Tracers for alpha weight (tracer_A, tracer_B).
        beta_tracers : tuple, optional
            Tracers for beta weight (tracer_X, tracer_Y).
        power_type : str, optional
            Type of power spectrum to use: 'total' or 'signal'.
        case : str, optional
            Case to use: 'variance' or 'projection'.
            
        Returns
        -------
        function : vegas.batchintegrand
            Vegas batch integrand function.
        """

        @vegas.batchintegrand
        def _integrand(x):
            # Extract mu and q from the integration variables
            mu = x[:, 0]
            q = x[:, 1]
            
            # Calculate integrand for each point
            if case == "variance":
                result = self.integrand_vec(
                        q, mu, K, alpha, beta,
                        alpha_tracers, beta_tracers,
                        power_type)
            elif case == "projection":
                result = self.integrand_projection(
                        q, mu, K, alpha, beta,
                        alpha_tracers,
                        power_type)
            else:
                raise ValueError(f"Invalid case: {case}")
            return result
        
        return _integrand

    
    def estimator(self, case, alpha, beta, K, minq, maxq, 
                           alpha_tracers=('m', 'm'), beta_tracers=None,
                           power_type='total', nitn=10, neval=1000, show_progress=True):
        """
        Implement the quadratic formula using Vegas Monte Carlo integration.
        
        This method is vectorized over K only.
        
        Parameters
        ----------
        alpha : str
            Key for first F function.
        beta : str
            Key for second F function.
        K : float or array-like
            Wavenumber(s).
        minq : float
            Minimum integration limit.
        maxq : float
            Maximum integration limit.
        alpha_tracers : tuple, optional
            Tracers for alpha weight (tracer_A, tracer_B).
        beta_tracers : tuple, optional
            Tracers for beta weight (tracer_X, tracer_Y).
        power_type : str, optional
            Type of power spectrum to use: 'total' or 'signal'.
        nitn : int, optional
            Number of iterations for Vegas.
        neval : int, optional
            Number of evaluations per iteration.
        show_progress : bool, optional
            Whether to show a progress bar during calculation.
            
        Returns
        -------
        result : float or array-like
            Result of the quadratic formula.
        error : float or array-like
            Estimated error of the result.
        """

        assert case in ["variance", "projection"], "Invalid case"
        if case == "variance":
            assert beta_tracers is not None, "beta_tracers must be provided for variance estimator"

        try:
            import vegas
        except ImportError:
            raise ImportError("The 'vegas' package is required for Monte Carlo integration. "
                             "Install it with 'pip install vegas'.")
        
        # Handle scalar or array inputs for K
        scalar_input = np.isscalar(K)
        K_array = np.atleast_1d(K)
        
        # Initialize result arrays
        means = np.zeros(len(K_array))
        errors = np.zeros(len(K_array))
        
        # Set up progress bar if requested
        iterator = tqdm(enumerate(K_array), total=len(K_array), 
                       desc=f"Computing variance for {alpha}{beta}") if show_progress else enumerate(K_array)
        
        # Calculate for each K value
        for i, k_val in iterator:
            if not show_progress:
                print(f"Computing for K = {k_val:.4f} ({i+1}/{len(K_array)})")
            
            # Get the batch integrand function
            integrand_func = self._outer_integral_vegas(
                k_val, alpha, beta, alpha_tracers, beta_tracers, power_type, case = case
            )
            
            # Set up the Vegas integrator with direct mu and q ranges
            integ = vegas.Integrator([[-1, 1], [minq, maxq]], nhcube_batch=1000)
            
            # Warm up the integrator
            integ(integrand_func, nitn=nitn//2, neval=neval//2)
            
            # Perform the integration
            result = integ(integrand_func, nitn=nitn, neval=neval)
            
            means[i] = result.mean
            errors[i] = result.sdev
        
        # Return scalar if input was scalar
        if scalar_input:
            return means[0], errors[0]
        
        return means, errors