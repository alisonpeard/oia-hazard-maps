import numpy as np
from scipy.stats import poisson
from scipy.stats import binom
from scipy.special import expit

def nll_poi(params, y, X):
    beta0, beta1 = params
    lambda_ = np.exp(beta0 + beta1 * X)
    return -np.sum(poisson.logpmf(y, lambda_))

def nll_binom(params, y, X, n=365):
    beta0, beta1 = params
    p = expit(beta0 + beta1 * X)
    return -np.sum(binom.logpmf(y, n, p))


def nll_trend(params, y, X, model="binom"):
    if model == "binom":
        return nll_binom(params, y, X)
    elif model == "poi":
        return nll_poi(params, y, X)


def mean_poi(t, params):
    beta0, beta1 = params
    return np.exp(beta0 + beta1 * t)

def mean_binom(t, params, n=365):
    beta0, beta1 = params
    p = expit(beta0 + beta1 * t)
    return n * p


def mean_trend(t, params, model="binom", n=365):
    if model == "binom":
        return mean_binom(t, params, n)
    elif model == "poi":
        return mean_poi(t, params)