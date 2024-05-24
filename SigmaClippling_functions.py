from astropy.stats import sigma_clip
import numpy as np

def dist_euclidiana_min(mo, po, m_X, evaluate):
    """
    Calculate the minimum Euclidean distance between observed data and the evaluated model.

    This function computes the Euclidean distance for each observed point to the model curve
    and finds the minimum distance for each point.

    Parameters:
    mo (array): Observed magnitudes.
    po (array): Observed phases.
    m_X (array): Phases at which the model is evaluated.
    evaluate (array): Model evaluated magnitudes at phases m_X.

    Returns:
    dist_list (list): List of minimum distances for each observed point.
    index_list (list): List of indices in m_X where the minimum distance occurs.
    original_index_list (list): List of original indices of observed points.
    """
    dist_list, index_list, original_index_list = [], [], []
    for i, (m, p) in enumerate(zip(mo, po)):
        # Compute the Euclidean distance for each point to the model curve
        dist = np.sqrt((m - evaluate)**2 + (p - m_X)**2)
        index = np.argmin(dist)
        dist_list.append(dist[index])
        index_list.append(index)
        original_index_list.append(i)
    return dist_list, index_list, original_index_list
    
def ell(evaluate, m_X, index_list):
    """
    Calculate a normalized chord length along the model curve.

    This function computes a normalized measure of the chord length along the curve at each point,
    which is useful for understanding how far along the curve a particular point lies.

    Parameters:
    evaluate (array): Model evaluated magnitudes at phases m_X.
    m_X (array): Phases at which the model is evaluated.
    index_list (list): List of indices in m_X corresponding to minimum distances.

    Returns:
    ell_list (list): List of normalized chord lengths for each index in index_list.
    """
    ell_list = []
    dev = np.gradient(evaluate, m_X)  # Compute the derivative of the model

    # Calculate the total length of the curve
    term = np.sqrt((dev**2) + 1)
    l_total = np.trapz(term, m_X)  # Integrate to find total length

    # Calculate normalized length for each index
    for i in index_list:
        if i != 0:
            term_range = np.sqrt((dev[:i+1]**2) + 1)
            l = np.trapz(term_range, m_X[:i+1])
            ell_list.append(l / l_total)  # Normalize the length
        else:
            ell_list.append(0)
    return ell_list

def sigmas(evaluate, m_X, index_list, err):
    """
    Calculate the sigma values for longitudinal and transverse errors.

    This function computes the standard deviation (sigma) values for the errors in the longitudinal
    (along the curve) and transverse (perpendicular to the curve) directions based on the gradient 
    of the evaluated model.

    Parameters:
    evaluate (array): Model evaluated magnitudes at phases m_X.
    m_X (array): Phases at which the model is evaluated.
    index_list (list): List of indices representing minimum distances from observed points to the model.
    err (array): Observed errors for each point.

    Returns:
    sigma_l (array): Longitudinal errors.
    sigma_d (array): Transverse errors.
    """
    sigma_l, sigma_d = [], []
    m = np.gradient(evaluate, m_X)  # Compute the gradient of the model

    for idx in index_list:
        theta = np.arctan(m[idx]) - (np.pi / 2)
        sigma_l.append(err[idx] * np.cos(theta))
        sigma_d.append(err[idx] * np.sin(theta))

    return np.abs(sigma_l), np.abs(sigma_d)

def remove_iterativo(dist, err_d, obs, phase, err, t, s, max_iter):
    """
    Iteratively apply sigma clipping to remove outliers based on Euclidean distance.

    This function applies sigma clipping in an iterative manner to the calculated distances
    between observed data and the model. In each iteration, data points identified as outliers
    are removed, and the process is repeated.

    Parameters:
    dist (array): Distances from observed points to the model curve.
    err_d (array): Transverse errors for each point.
    obs (array): Observed magnitudes.
    phase (array): Observed phases.
    err (array): Observed errors for each point.
    t (array): Observation times.
    s (float): Sigma value for clipping.
    max_iter (int): Maximum number of iterations for sigma clipping.

    Returns:
    Tuple containing arrays of clipped and outlier data for magnitudes, phases, errors, and times.
    """
    obs_clipped, phase_clipped, err_clipped, t_clipped = obs.copy(), phase.copy(), err.copy(), t.copy()
    obs_out, phase_out, err_out, t_out = [], [], [], []

    for iteration in range(max_iter):
        clipped_dist = sigma_clip(dist, sigma=s, maxiters=1, masked=True)
        obs_temp, phase_temp, err_temp, t_temp, dist_temp = [], [], [], [], []

        for idx, val in enumerate(dist):
            if not clipped_dist.mask[idx]:
                obs_temp.append(obs_clipped[idx])
                phase_temp.append(phase_clipped[idx])
                err_temp.append(err_clipped[idx])
                t_temp.append(t_clipped[idx])
                dist_temp.append(dist[idx])
            else:
                obs_out.append(obs_clipped[idx])
                phase_out.append(phase_clipped[idx])
                err_out.append(err_clipped[idx])
                t_out.append(t_clipped[idx])

        obs_clipped, phase_clipped, err_clipped, t_clipped, dist = obs_temp, phase_temp, err_temp, t_temp, dist_temp

        if not clipped_dist.mask.any():
            break  # Stop if no more outliers are found

    return obs_clipped, phase_clipped, err_clipped, t_clipped, obs_out, phase_out, err_out, t_out