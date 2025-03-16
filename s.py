from sympy import symbols, sqrt, integrate, legendre, simplify, expand, series, collect, limit, Symbol, Poly, together, apart, factor

def calculate_squeezed_limit(F_function, max_power=2, take_squeezed_limit=True, Ma = 1, Mb = 1, K = None, k = None, mu = None, P_k = None, Q_k = None):
    """
    Calculate the squeezed limit (K→0) for a general kernel function F.
    
    Parameters:
    -----------
    F_function : callable
        A function that takes (k1, k2, k1_mag, k2_mag, dot_product) and returns the kernel value
    max_power : int
        Maximum power of K/k to expand to
        
    Returns:
    --------
    The squeezed limit expression
    """
    # Define symbols
    #K, k, mu = symbols('K k mu')
    #P_k = symbols('P_k')  # Power spectrum
    #Q_k = symbols('Q_k')  # d ln P / d ln k
    small_param = K/k
    
    # Define Legendre polynomials
    P0 = 1
    P1 = mu
    P2 = (3*mu**2 - 1)/2
    
    # Case 1: F(K, k - K)
    k_minus_K_mag = sqrt(k**2 - 2*k*K*mu + K**2)
    K_dot_k_minus_K = K*(k*mu - K)
    
    F_case1 = Ma*F_function(K, None, K, k_minus_K_mag, K_dot_k_minus_K)
    
    # Case 2: F(K, -k)
    F_case2 = Mb*F_function(K, None, K, k, -K*k*mu)
    
    # Expand for small K/k
    try:
        # For Case 1 with sqrt term, expand carefully
        F_case1_subst = F_case1.subs(k, K/small_param)
        F_case1_series = series(F_case1_subst, small_param, 0, max_power).removeO()
        F_case1_expanded = F_case1_series.subs(small_param, K/k)
        F_case1_expanded = expand(F_case1_expanded)
        F_case1_expanded = collect(F_case1_expanded, K/k)
        
        # Express in terms of Legendre polynomials
        F_case1_legendre = F_case1_expanded.subs(mu**2, (2*P2 + 1)/3)
        F_case1_legendre = expand(F_case1_legendre)
        F_case1_legendre = collect(F_case1_legendre, [P0, P1, P2])
        
        # For Case 2
        F_case2_subst = F_case2.subs(k, K/small_param)
        F_case2_series = series(F_case2_subst, small_param, 0, max_power).removeO()
        F_case2_expanded = F_case2_series.subs(small_param, K/k)
        F_case2_expanded = expand(F_case2_expanded)
        F_case2_expanded = collect(F_case2_expanded, K/k)
        
        # Express in terms of Legendre polynomials
        F_case2_legendre = F_case2_expanded.subs(mu, P1)
        F_case2_legendre = collect(F_case2_legendre, [P0, P1, P2])
        
        # P(|K-k|) ≈ P(k)(1 - μ(K/k)Q(k))
        P_K_minus_k = P_k * (1 - mu * (K/k) * Q_k)
        
        # Combine the terms
        combination = 2 * (F_case1_legendre * P_K_minus_k + F_case2_legendre * P_k)
        combination = expand(combination)
        combination = collect(combination, [P0, P1, P2])

        combination_rational = together(combination)
        combination_parts = apart(combination_rational, K)

        terms = combination_parts.as_ordered_terms()
        inverse_K_terms = []
        regular_terms = []
        
        for term in terms:
            if 1/K in term.as_ordered_factors():
                inverse_K_terms.append(term)
            else:
                regular_terms.append(term)
        
        # Take the limit of only the regular terms
        if regular_terms:
            regular_part = sum(regular_terms)
            if take_squeezed_limit:
                regular_limit = limit(regular_part, K, 0)
            else:
                regular_limit = regular_part
        else:
            regular_limit = 0

         # Combine the results
        if inverse_K_terms:
            inverse_K_part = sum(inverse_K_terms)
            result = inverse_K_part + regular_limit
        else:
            result = regular_limit
        
        return result
        
    except Exception as e:
        return f"Error in calculation: {e}"
    
def integrate_over_mu(expr):
    return integrate(expr, (mu, -1, 1))

# Example usage with F_S from your code
def F_S(k1, k2, k1_mag, k2_mag, dot_product):
    return (1/2) * (1/k1_mag**2 + 1/k2_mag**2) * dot_product


def F_T(k1, k2, k1_mag, k2_mag, dot_product):
    return dot_product**2/(k1_mag**2*k2_mag**2)-1/3

def F_G(k1, k2, k1_mag, k2_mag, dot_product):
    return k1_mag/k1_mag


take_squeezed_limit = True

K = symbols('K')
k = symbols('k')
mu = symbols('mu')
P_k = symbols('P_k')
Q_k = symbols('Q_k')

# Calculate the squeezed limit for F_S
result_S = calculate_squeezed_limit(F_S, take_squeezed_limit=take_squeezed_limit, K = K, k = k, mu = mu, P_k = P_k, Q_k = Q_k)
print("Squeezed limit for F_S:")
print(result_S)

Ma, Mb = symbols('Ma Mb')
result_S_new = calculate_squeezed_limit(F_S, take_squeezed_limit=take_squeezed_limit, Ma = Ma, Mb = Mb, K = K, k = k, mu = mu, P_k = P_k, Q_k = Q_k)
print("Squeezed limit for F_S new:")
print(result_S_new)

# Calculate the squeezed limit for F_T
result_T = calculate_squeezed_limit(F_T, take_squeezed_limit=take_squeezed_limit, K = K, k = k, mu = mu, P_k = P_k, Q_k = Q_k)
print("Squeezed limit for F_T:")
print(result_T)

# Calculate the squeezed limit for F_G
result_G = calculate_squeezed_limit(F_G, take_squeezed_limit=take_squeezed_limit, K = K, k = k, mu = mu, P_k = P_k, Q_k = Q_k)
print("Squeezed limit for F_G:")
print(result_G)


from sympy import symbols, sqrt, integrate, legendre, simplify, expand, series, collect, limit, Symbol, Poly, together, apart, factor, nsimplify


from sympy import symbols, sqrt, integrate, legendre, simplify, expand, series, collect, limit, Symbol, Poly, together, apart, factor

def calculate_squeezed_limit_with_integration(F1_function, F2_function=None, max_power=2, take_squeezed_limit=True, Ma1=1, Mb1=1, Ma2=1, Mb2=1, K = None, k = None, mu = None, P_k = None, Q_k = None):
    """
    Calculate the squeezed limit for one or two kernel functions, multiply them,
    integrate over mu and k with the form:
    ∫dk k² ∫dμ combined_squeezed/2 * (PAA(k) * PBB(|K-k|))
    
    Parameters:
    -----------
    F1_function : callable
        First kernel function
    F2_function : callable or None
        Second kernel function (if None, only F1 is used)
    max_power : int
        Maximum power of K/k to expand to
    take_squeezed_limit : bool
        Whether to take the limit K→0 for regular terms
    Ma1, Mb1 : symbols or numbers
        Multipliers for the two cases of F1
    Ma2, Mb2 : symbols or numbers
        Multipliers for the two cases of F2
        
    Returns:
    --------
    The integrated result grouped by power spectrum terms
    """
    # Define symbols
    """K, k, mu = symbols('K k mu')
    P_k = symbols('P_k')  # Linear power spectrum
    Q_k = symbols('Q_k')  # d ln P / d ln k"""
    
    # Calculate squeezed limit for F1
    squeezed_limit_F1 = calculate_squeezed_limit(F1_function, max_power, take_squeezed_limit, Ma1, Mb1, K, k, mu, P_k, Q_k)
    
    # If F2 is provided, calculate its squeezed limit and multiply
    if F2_function is not None:
        squeezed_limit_F2 = calculate_squeezed_limit(F2_function, max_power, take_squeezed_limit, Ma2, Mb2, K, k, mu, P_k, Q_k)
        print("Print", squeezed_limit_F2)
        combined_squeezed = squeezed_limit_F1 * squeezed_limit_F2
    else:
        combined_squeezed = squeezed_limit_F1
    
    # Expand the combined result
    combined_squeezed = expand(combined_squeezed)
    
    # Define bias and shot noise terms
    bAA, bBB = symbols('b_AA b_BB')
    sA, sB = symbols('s_A s_B')
    Pnlin = symbols('P_nlin')  # Non-linear power spectrum
    Q_nlin = symbols('Q_nlin')  # d ln Pnlin / d ln k


    nhalo = float(values['survey_config']['tracer_properties']['nhalo'])
    shot_noise = 1.0 / nhalo

    sA, sB = 0, 0
    
    # Define the power spectra with bias and shot noise
    PAA = bAA**2 * Pnlin + sA
    PBB = bBB**2 * Pnlin + sB
    
    # For PBB(|K-k|), expand around k for small K
    # PBB(|K-k|) ≈ PBB(k) * (1 - μ(K/k)Q_nlin(k))
    PBB_K_minus_k = bBB**2 * Pnlin * (1 - mu * (K/k) * Q_nlin) + sB
    
    # Construct the integrand: combined_squeezed/2 * PAA(k) * PBB(|K-k|)
    integrand = combined_squeezed/2 /PAA / PBB_K_minus_k
    
    # Integrate over mu
    mu_integrated = integrate(integrand, (mu, -1, 1))

    mu_integrated = simplify(mu_integrated)
    
    # Multiply by k² for the k integration
    k_integrand = k**2 * mu_integrated
    
    # Expand and collect terms
    k_integrand = expand(k_integrand)
    
    # Collect terms by powers of k to prepare for k integration
    k_integrand = collect(k_integrand, k)
    
    # We don't actually perform the k integration symbolically as it depends on the 
    # specific form of Pnlin(k), but we organize the terms for easier integration
    
    # Collect terms by Pnlin, shot noise, and their combinations
    result = collect(k_integrand, [Pnlin**2, Pnlin*sA, Pnlin*sB, sA*sB])

    if take_squeezed_limit:
        result = limit(result, K, 0)
    else:
        result = result
    
    return result.simplify()


def calculate_squeezed_limit_with_integration_(F1_function, F2_function=None, max_power=2, take_squeezed_limit=True, Ma1=1, Mb1=1, Ma2=1, Mb2=1, K = None, k = None, mu = None, P_k = None, Q_k = None):
    """
    Calculate the squeezed limit for one or two kernel functions, multiply them,
    integrate over mu and k with the form:
    ∫dk k² ∫dμ combined_squeezed/2 * (PAA(k) * PBB(|K-k|))
    
    Parameters:
    -----------
    F1_function : callable
        First kernel function
    F2_function : callable or None
        Second kernel function (if None, only F1 is used)
    max_power : int
        Maximum power of K/k to expand to
    take_squeezed_limit : bool
        Whether to take the limit K→0 for regular terms
    Ma1, Mb1 : symbols or numbers
        Multipliers for the two cases of F1
    Ma2, Mb2 : symbols or numbers
        Multipliers for the two cases of F2
        
    Returns:
    --------
    The integrated result with terms separated by K power
    """
    # Define symbols
    """K, k, mu = symbols('K k mu')
    P_k = symbols('P_k')  # Linear power spectrum
    Q_k = symbols('Q_k')  # d ln P / d ln k"""
    
    # Calculate squeezed limit for F1
    squeezed_limit_F1 = calculate_squeezed_limit(F1_function, max_power, take_squeezed_limit, Ma1, Mb1, K = K, k = k, mu = mu, P_k = P_k, Q_k = Q_k)
    
    # If F2 is provided, calculate its squeezed limit and multiply
    if F2_function is not None:
        squeezed_limit_F2 = calculate_squeezed_limit(F2_function, max_power, take_squeezed_limit, Ma2, Mb2, K = K, k = k, mu = mu, P_k = P_k, Q_k = Q_k)
        print("Print", squeezed_limit_F2)
        combined_squeezed = squeezed_limit_F1 * squeezed_limit_F2
    else:
        combined_squeezed = squeezed_limit_F1
    
    # Expand the combined result
    combined_squeezed = expand(combined_squeezed)
    
    # Define bias and shot noise terms
    bAA, bBB = symbols('b_AA b_BB')
    sA, sB = symbols('s_A s_B')
    Pnlin = symbols('P_nlin')  # Non-linear power spectrum
    Q_nlin = symbols('Q_nlin')  # d ln Pnlin / d ln k

    sA, sB = 0, 0
    
    # Define the power spectra with bias and shot noise
    PAA = bAA**2 * Pnlin + sA
    PBB = bBB**2 * Pnlin + sB
    
    # For PBB(|K-k|), expand around k for small K
    # PBB(|K-k|) ≈ PBB(k) * (1 - μ(K/k)Q_nlin(k))
    PBB_K_minus_k = bBB**2 * Pnlin * (1 - mu * (K/k) * Q_nlin) + sB
    
    # Construct the integrand: combined_squeezed/2 * PAA(k) * PBB(|K-k|)
    integrand = combined_squeezed/2 /PAA / PBB * (1 + mu*(K/k) * Q_nlin)
    
    # Integrate over mu
    mu_integrated = integrate(integrand, (mu, -1, 1))
    print("MU INTEGRATED", mu_integrated)
    mu_integrated = simplify(mu_integrated)
    
    # Multiply by k² for the k integration
    k_integrand = k**2 * mu_integrated
    
    # Expand to get individual terms
    result_expanded = expand(k_integrand)
    print("EXPANDED RESULT:", result_expanded)
    
    # Split terms based on power of K
    def separate_terms_by_K_power(expr):
        """
        Separate terms into those with negative powers of K and those with non-negative powers
        """
        # Get individual terms
        terms = expr.as_ordered_terms()
        
        negative_power_terms = 0
        non_negative_power_terms = 0
        
        for term in terms:
            # Method 1: Try to determine minimum power of K in the term
            try:
                # Convert to a rational function of K
                rat_func = together(term)
                
                # Get numerator and denominator
                num, den = rat_func.as_numer_denom()
                
                # See if K appears in denominator with higher power than numerator
                if den.has(K):
                    # Get the power of K in denominator
                    try:
                        # This works for simple cases like K^n
                        den_poly = Poly(den, K)
                        num_poly = Poly(num, K) if num.has(K) else Poly(0, K)
                        
                        # Get highest powers
                        den_degree = den_poly.degree()
                        num_degree = num_poly.degree() if num.has(K) else 0
                        
                        # If denominator power > numerator power, it's a negative power
                        if den_degree > num_degree:
                            negative_power_terms += term
                        else:
                            non_negative_power_terms += term
                    except:
                        # If we can't determine polynomial degrees, try method 2
                        term_times_K = term * K
                        
                        # Check if multiplying by K makes the term K-free
                        if not term_times_K.has(K):
                            # This is a 1/K term
                            negative_power_terms += term
                        else:
                            # Try one more test: multiply by large power of K and take limit
                            test_term = term * K**10
                            try:
                                lim = limit(test_term, K, 0)
                                # If limit is 0, then original term has negative power of K
                                if lim == 0:
                                    negative_power_terms += term
                                else:
                                    non_negative_power_terms += term
                            except:
                                # If limit fails, use string-based approach as fallback
                                term_str = str(term)
                                if '/K' in term_str:
                                    negative_power_terms += term
                                else:
                                    non_negative_power_terms += term
                else:
                    # No K in denominator, so power is non-negative
                    non_negative_power_terms += term
            except Exception as e:
                # If rational function conversion fails, use a simpler approach
                # Check if term contains 1/K pattern in string representation
                term_str = str(term)
                if '/K' in term_str:
                    # Further check: multiply by K and see if K remains
                    try:
                        if (term * K).has(K):
                            # K still remains, power is less than -1
                            negative_power_terms += term
                        else:
                            # Exactly 1/K term
                            negative_power_terms += term
                    except:
                        # Can't determine precisely, assume negative
                        negative_power_terms += term
                else:
                    # Probably non-negative power
                    non_negative_power_terms += term
                
        return negative_power_terms, non_negative_power_terms
    
    # Separate terms
    negative_power_terms, non_negative_power_terms = separate_terms_by_K_power(result_expanded)
    
    print("NEGATIVE POWER TERMS:", negative_power_terms)
    print("NON-NEGATIVE POWER TERMS:", non_negative_power_terms)
    
    # Take the limit of only the non-negative power terms
    if take_squeezed_limit and non_negative_power_terms != 0:
        try:
            non_negative_limit = limit(non_negative_power_terms, K, 0)
        except Exception as e:
            print(f"Warning: Could not take limit of non-negative terms: {e}")
            non_negative_limit = non_negative_power_terms
    else:
        non_negative_limit = non_negative_power_terms
    
    # Combine the negative power terms (unchanged) with the limit of non-negative terms
    final_result = negative_power_terms + non_negative_limit
    
    # Convert any floating-point numbers to rationals for cleaner output
    final_result = nsimplify(final_result, rational=True)
    
    return final_result.simplify()


def final_integrand(result):
    pi = symbols('pi')
    factor = 2*pi/(2*pi)**3*result
    return factor


