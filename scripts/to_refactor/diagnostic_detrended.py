"""
To do:
- add diagnostic stability plots
- run for several models, pixels, scenarios
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from glob import glob
import os
import random
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import logging
from scipy.stats import genpareto
from scipy.optimize import minimize
from scipy.special import expit
import stats
from tqdm import tqdm


def dgpd_pmf(k, scale, shape):
    """
    Discrete Generalized Pareto Distribution PMF
    k: integer values (0, 1, 2, ...)
    scale: scale parameter (σ)
    shape: shape parameter (ξ)

    From https://arxiv.org/pdf/1707.05033
    """
    if shape == 0:
        # top of page 4
        p = 1 - np.exp(-1/scale)
        return p * np.exp(-k/scale)
    else:
        # Eq. (4)
        k_lower = 1 + shape * k / scale
        k1_lower = 1 + shape * (k + 1) / scale
        if k_lower <= 0 or k1_lower <= 0:
            return 0.0
        
        gpd_k = (1 + shape * k / scale)**(-1/shape)
        gpd_k1 = (1 + shape * (k + 1) / scale)**(-1/shape)
        return gpd_k - gpd_k1


def dgpd_survival(k, scale, shape):
    """Survival function for D-GPD. This is the same 
    as the GPD."""
    if shape == 0:
        return np.exp(-k/scale)
    else:
        return (1 + shape * k / scale)**(-1/shape)


def dgpd_quantile(p, scale, shape):
    """
    Quantile function for D-GPD (return level)
    p: exceedance probability (e.g., 1/return_period)
    """
    if shape == 0:
        x = -scale * np.log(p)
    else:
        x = (scale / shape) * (p**(-shape) - 1)
    return np.floor(x) # only floor if needd


def nll_dgpd(params, data):
    """Negative log-likelihood for D-GPD"""
    scale, shape = params
    
    if scale <= 0:
        return np.inf
    
    log_probs = []
    for k in data:
        pmf_val = dgpd_pmf(k, scale, shape)
        if pmf_val <= 0:
            return np.inf
        log_probs.append(np.log(pmf_val))
    
    return -np.sum(log_probs)


temperature = 35
input_dir = "../annual_stats"
scenario = ["rcp26", "rcp45", "rcp85"][2]

epochs = [2030, 2050, 2080]
rps = [5, 10, 20, 50, 100, 200, 500, 1000]

ignore = ["DS_Store"]

model_idx = 1

if __name__ == "__main__":

    output_dir = "../diagnoistics/hazard_detrended"
    os.makedirs(output_dir, exist_ok=True)
    
    scenario_dir = os.path.join(input_dir, scenario)

    models = [d for d in os.listdir(scenario_dir) if d not in ignore]
    if len(models) == 0:
        raise FileExistsError(f"No models found for scenario {scenario}")
    else:
        print(f"Found {len(models)} models for scenario {scenario}")

    model = models[model_idx]
    model_dir = os.path.join(scenario_dir, model)
    files = glob(f"{model_dir}/*.nc")
    if len(files) == 0:
        raise FileNotFoundError(f"No files found for model {model}, scenario {scenario}")

    ds = xr.open_mfdataset(files, engine="netcdf4")
    ds = ds.load()

    print(f"Processing model: {model}")

    x = ds[f"hd{temperature}"].values
    n, h, w = x.shape

    years = ds["year"].values

    lat = random.randint(0, h-1)
    lon = random.randint(0, w-1)
    # lat, lon = 14, 10

    data = x[:, lat, lon]
    if (data == -9999).all() or np.isnan(data).all():
        raise ValueError(f"All data missing at lat {lat}, lon {lon}")

    # Create mask before filtering data
    valid_mask = (data != -9999) & ~np.isnan(data)
    years_valid = years[valid_mask]
    data_valid = data[valid_mask]

    """
    Part 1: Detrend the data
    """
    # estimate initial parameters for trend model
    p_init = np.clip(data_valid.mean() / 365, 0.01, 0.99)  # avoid extremes
    beta0_init = np.log(p_init / (1 - p_init))  # logit of mean probability
    years_centered = years_valid - years_valid.mean()

    # fit the trend model
    fit_trend = minimize(
        stats.nll_binom,
        [p_init, 0],
        args=(data_valid, years_centered),
        method='L-BFGS-B',
        bounds=[(-10, 10), (-0.1, 0.1)]
    )
    if not fit_trend.success:
        raise RuntimeError(f"Trend fitting failed at lat {lat}, lon {lon}: {fit_trend.message}")

    beta0_trend, beta1_trend = fit_trend.x
    print(f"Intercept: {beta0_trend:.4f}, Slope: {beta1_trend:.4f}")
    
    expected = stats.mean_binom(years_centered, (beta0_trend, beta1_trend))
    detrended = data_valid - expected
    detrended = np.maximum(detrended, 0)

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(years_valid, data_valid, color="blue", label="data")
    ax.plot(years_valid, expected, color='red', label="expected")
    ax.plot(years_valid, detrended, 'purple', label="residuals")
    ax.legend(loc="lower left", fontsize=6, framealpha=0.0)
    ax.set_title(f"hd{temperature} at lat {lat}, lon {lon}")
    plt.show()

    # %%
    """
    Part 2: fit stationary d-GPD to detrended data.
    """
    # initialize arrays with NaNs
    quantiles = np.linspace(70.0, 92.0, 25)
    nq = len(quantiles)
    locs = np.full(nq, np.nan)
    shapes = np.full(nq, np.nan)
    scales = np.full(nq, np.nan)
    nexceeds = np.full(nq, np.nan)
    return_levels = np.full((len(rps), len(epochs), nq), np.nan)

    B = 30
                
    for iq, quantile in enumerate(quantiles):
        print(f"\nChosen threshold at lat {lat}, lon {lon}: {quantile}")

        # detrended_b = np.random.choice(detrended, size=len(detrended), replace=True)
        threshold = np.percentile(detrended, quantile)

        exceedances = detrended[detrended > threshold] - threshold
        exceed_years = years_valid[detrended > threshold]

        nexceeds[iq] = len(exceedances)
        if nexceeds[iq] < 5:
            print(f"WARNING: Insufficient exceedances at lat {lat}, lon {lon}")
            continue
        # print(f"Number of exceedances at lat {lat}, lon {lon}: {len(exceedances)}\n")
        
        # initial estimates using normal GPD
        init_params = genpareto.fit(exceedances, floc=0)
        scale_init = init_params[2]
        shape_init = init_params[0]
        init_params = [scale_init, shape_init]
                
        if scale_init <= 0:
            print(f"WARNING: Initial guess for scale is non-positive at lat {lat}, lon {lon}")

        result = minimize(
            nll_dgpd,
            init_params,
            args=(exceedances,),
            method='L-BFGS-B',
            bounds=[(2e-5, None), (-0.5, 0.5)]
        )
                
        if not result.success:
            print(f"WARNING: Optimization failed at lat {lat}, lon {lon}: {result.message}")
        
        scale, shape = result.x  
    
        locs[iq] = threshold                      
        shapes[iq] = shape
        scales[iq] = scale


        # get return levels 
        for i_epoch, epoch in enumerate(epochs):
            expected = stats.mean_binom(epoch - years_valid.mean(), (beta0_trend, beta1_trend))
            for rp_idx, T in enumerate(rps):
                return_level = expected + threshold + dgpd_quantile(1/T, scale, shape)
                return_levels[rp_idx, i_epoch, iq] = return_level # min(return_level, 365)         
    # %%
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharex=True)
    axs[0].plot(quantiles, locs)
    axs[1].plot(quantiles, scales)
    axs[2].plot(quantiles, shapes)
    axs[3].plot(quantiles, nexceeds)
    axs[0].set_title("Threshold (loc)")
    axs[1].set_title("Scale")
    axs[2].set_title("Shape")
    axs[3].set_title("Number of exceedances")
    # %%
    cmap = plt.get_cmap("viridis_r")

    fig, axs = plt.subplots(1, len(epochs), figsize=(10, 10), sharex=True, sharey=True)
    for i, ax in enumerate(axs):
        epoch = epochs[i]
        for j, q in enumerate(quantiles):
            color = cmap(j / len(quantiles))
            ax.plot(rps, return_levels[:, i, j], color=color, label=f"q={q:.1f}")
        ax.set_xscale("log")
        ax.set_title(f"Epoch: {epoch}")
        ax.set_xlabel("Return period (years)")
        if epoch == epochs[0]:
            ax.legend(loc="upper left", fontsize=6, framealpha=0.0)
    axs[0].set_ylabel(f"Return level hd{temperature} (days)")
    plt.suptitle(f"Model: {model}, Scenario: {scenario}, Lat: {lat}, Lon: {lon}")
    plt.tight_layout()
    plt.show()
# %%
