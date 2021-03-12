import pandas as pd
import numpy as np
import numba


@numba.jit
def _regress(y, X, multiplier=1):
    ones = np.ones(len(X)) * multiplier
    A = np.vstack((ones, X)).T
    output = np.linalg.inv(A.T @ A) @ A.T @ y
    return output


@numba.jit
def _ar_1(residuals):
    lagged_endogenous = np.roll(residuals, 1)[1:]
    endogenous = residuals[1:]
    ar_coefficients = _regress(endogenous, lagged_endogenous)
    a, b = ar_coefficients
    errors = endogenous - (b * lagged_endogenous + a)
    var = errors.var()
    return a, b, var


@numba.jit
def _compute_ou_parameters(a, b, var):
    k = -np.log(b) * 252
    m = a / (1 - b)
    # sigma = np.sqrt((var * 2 * k) / (1 - b ** 2))
    sigma_eq = np.sqrt(var / (1 - b ** 2))
    # tau = 1 / k
    return k, m, sigma_eq


@numba.jit
def auto_corr(x):
    return np.corrcoef(x[:-1], x[1:])[0, 1]


@numba.jit
def _alternative_compute_out_parameters(res):
    cum_res = res.cumsum()
    b = auto_corr(cum_res)
    k = -np.log(b) * 252
    temp = cum_res[1:] - np.roll(cum_res, 1)[1:] * b
    a = temp.mean()
    cosine = temp - a
    m = a / (1 - b)
    sigma_eq = np.sqrt(cosine.var() / (1 - b ** 2))
    return k, m, sigma_eq


@numba.jit
def _fit_ornstein_uhlenbeck(y_arr, x_arr):
    intercept_arr = []
    slope_arr = []

    a_arr = []
    b_arr = []
    var_arr = []

    k_arr = []
    m_arr = []
    # sigma_arr = []
    sigma_eq_arr = []
    # tau_arr = []

    if y_arr.shape != x_arr.shape:
        raise IndexError("Arrays do not align")
    for i in range(y_arr.shape[1]):
        y = y_arr[:, i]
        X = x_arr[:, i]
        output = _regress(y, X)
        intercept, slope = output

        residuals = y - (slope * X + intercept)

        # Primary Method
        residuals = residuals.cumsum()
        ar_result = _ar_1(residuals)
        a, b, errors_var = ar_result
        ou_parameters = _compute_ou_parameters(a, b, errors_var)

        # Alternative method
        # ar_result = _ar_1(residuals.cumsum())
        # a, b, errors_var = ar_result
        # ou_parameters = _alternative_compute_out_parameters(residuals)

        k, m, sigma_eq = ou_parameters

        intercept_arr.append(intercept)
        slope_arr.append(slope)

        a_arr.append(a)
        b_arr.append(b)
        var_arr.append(errors_var)

        k_arr.append(k)
        m_arr.append(m)
        # sigma_arr.append(sigma)
        sigma_eq_arr.append(sigma_eq)
        # tau_arr.append(tau)

    return intercept_arr, slope_arr, a_arr, b_arr, var_arr, k_arr, m_arr, sigma_eq_arr


def fit_ou_process(stocks, benchmarks):
    ou_estimates = _fit_ornstein_uhlenbeck(stocks.values, benchmarks.values)
    ou_estimates = pd.DataFrame(ou_estimates,
                                index=['intercept', 'coefficient', 'a', 'b', 'error_var', 'k', 'm', 'sigma_eq'],
                                columns=stocks.columns
                                ).T
    ou_estimates.loc[:, 'tau'] = 1 / ou_estimates.loc[:, 'k']
    ou_estimates.loc[:, 'centered_m'] = ou_estimates.loc[:, 'm'] - ou_estimates.loc[:, 'm'].mean()
    ou_estimates.loc[:, 's_score'] = -ou_estimates.loc[:, 'centered_m'] / ou_estimates.loc[:, 'sigma_eq']
    return ou_estimates
