import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from sklearn.linear_model import HuberRegressor as HR
from sklearn.utils import resample
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

def error_HR(X, y, model):
    predictions = model.predict(X)
    MSE = mean_squared_error(y, predictions)

    # Asegurarse de que X es un DataFrame de pandas
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Agregar una columna constante para el intercepto si el modelo incluye uno
    if model.fit_intercept:
        X = pd.concat([pd.DataFrame({"Constant": np.ones(len(X))}), X], axis=1)

    # Calcular la matriz de covarianza
    var_b = MSE * (np.linalg.inv(np.dot(X.T, X)).diagonal())
    sd_b = np.sqrt(var_b)

    return pd.DataFrame({
        "Features": X.columns,
        "Coefficients": model.coef_,
        "Standard Errors": sd_b
    })

def coeff_errors(model, X, y):
    """
    Estimate the errors of the Fourier coefficients using bootstrapping.

    Parameters:
    model: The regression model used.
    X (array-like): The input features used for the model.
    y (array-like): The mag values.
    n_bootstraps (int): Number of bootstrap samples to use.

    Returns:
    errors (array): Estimated standard errors of the model coefficients.
    """
    n_bootstraps=100
    coefs = np.zeros((n_bootstraps, X.shape[1]))
    for i in range(n_bootstraps):
        X_sample, y_sample = resample(X, y)
        model_clone = clone(model)
        model_clone.fit(X_sample, y_sample)
        coefs[i, :] = model_clone.coef_

    errors = np.std(coefs, axis=0)
    return errors

def fourier(T0,period,times,mags,n):
    """
    Perform a Fourier fit to the observed magnitudes of a star.
    
    Parameters:
    T0 (float): Reference time.
    period (float): Period of the star.
    times (array): Array of observation times.
    mags (array): Array of observed magnitudes.
    n (int): Order of the Fourier series.
    
    Returns:
    model: Fitted Fourier model.
    coeff (array): Coefficients of the Fourier model.
    """
    ph1 = ((times - T0) / period) - np.floor((times - T0) / period)
    temp = [np.ones(len(ph1)).reshape(-1, 1), *[np.sin(2 * (i + 1) * np.pi * ph1).reshape(-1, 1) for i in range(n)], *[np.cos(2 * (j + 1) * np.pi * ph1).reshape(-1, 1) for j in range(n)]]
    cn = ['c'] + [f'a{i+1}' for i in range(n)] + [f'b{j+1}' for j in range(n)]

    df = pd.DataFrame(np.concatenate(temp, axis=1), columns=cn)
    model = HR(fit_intercept=False, max_iter=1000000)
    model.fit(df, mags)
    return model, model.coef_

def evaluar(T0,modelo,times,period,mags,n):
    """
    Evaluate the Fourier model at given times.

    This function generates a DataFrame based on the Fourier series and uses the model to predict magnitudes for given times.

    Parameters:
    T0 (float): Reference time, typically the time of maximum light.
    modelo: Trained Fourier model from the 'fourier' function.
    times (array): Times at which to evaluate the model.
    period (float): Period of the star.
    mags (array): Observed magnitudes, used only for array shaping.
    n (int): Order of the Fourier series.

    Returns:
    array: Predicted magnitudes at the specified times.
    """
    # Calculate the phase for each time point
    ph1 = ((times - T0) / period) - np.floor((times - T0) / period)
    # Generate Fourier series components
    components = [np.ones(len(ph1)).reshape(-1, 1), *[np.sin(2 * (i + 1) * np.pi * ph1).reshape(-1, 1) for i in range(n)], *[np.cos(2 * (j + 1) * np.pi * ph1).reshape(-1, 1) for j in range(n)]]
    # Column names for DataFrame
    cn1 = ['c'] + [f'a{i+1}' for i in range(n)] + [f'b{j+1}' for j in range(n)]
    # Create DataFrame with components
    df1 = pd.DataFrame(np.concatenate(components, axis=1), columns=cn1)

    # Use the model to predict magnitudes
    return modelo.predict(df1)