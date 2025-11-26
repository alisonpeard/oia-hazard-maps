"""
This gave unexpected results, so it's paused for now.

The major differences from the nonstationary implementaiton are
- fit a Poisson/Binomial GLM to the data to estimate the trend
- subtract the time-varying expected value to get a time series of residuals
- fit a nonstationary D-GPD to the residuals, with scale parameter varying with time
- compute return levels by adding back the time-varying expected value to the D-GPD quantiles
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from glob import glob
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from warnings import warn
from scipy.stats import genpareto
from scipy.optimize import minimize
import stats


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



def dgpd_quantile_nonstationary(p, beta0, beta1, shape, time):
    scale = np.exp(beta0 + beta1 * time)
    return dgpd_quantile(p, scale, shape)


def nonstationary_nll_dgpd(params, data, time):
    """Negative log-likelihood for D-GPD
    initial_guess = [np.log(scale_stationary_fit), 0.0, shape_stationary_fit]
    """
    beta0, beta1, shape = params

    scale_vector = np.exp(beta0 + beta1 * time)
        
    log_probs = []

    for k, scale_t in zip(data, scale_vector):
        if scale_t <= 0:
            return np.inf
        pmf_val = dgpd_pmf(k, scale_t, shape)
        if pmf_val <= 0:
            return np.inf
        log_probs.append(np.log(pmf_val))
    
    return -np.sum(log_probs)


window = 200 # 30
temperature = 39
input_dir = "annual_stats"
scenarios = ["rcp26", "rcp45", "rcp85"] #"historical"
rps = [5, 10, 20, 50, 100, 200, 500, 1000]
epochs = [2030, 2050, 2080]
ignore = ["DS_Store"]
dry_run = True


if __name__ == "__main__":
    output_dir = f"hazard_by_model/hd{temperature}"
    os.makedirs(output_dir, exist_ok=True)
    
    for scenario in scenarios:
        scenario_dir = os.path.join(input_dir, scenario)
        models = [d for d in os.listdir(scenario_dir) if d not in ignore]

        if len(models) == 0:
            warn(f"No models found for scenario {scenario}")
            continue

        for model in models:

            model_dir = os.path.join(scenario_dir, model)
            files = glob(f"{model_dir}/*.nc")
            if len(files) == 0:
                warn(f"No files found for model {model}, scenario {scenario}")
                continue
            ds = xr.open_mfdataset(files, engine="netcdf4")
            ds = ds.load()

            print(f"Processing model: {model}, scenario: {scenario}")

            # ds_epoch = ds.sel(year=slice(epoch - window/2, epoch + window/2 - 1))
            # ds_epoch = ds_epoch.load()
            ds_epoch = ds.copy()

            x = ds_epoch[f"hd{temperature}"].values
            n, h, w = x.shape

            years = ds_epoch["year"].values
            years_min = years.min()
            years_max = years.max()
            years = (years - years_min) / (years_max - years_min)

            # initialize arrays with NaNs
            tail_indices = np.full((h, w), np.nan)
            locs = np.full((h, w), np.nan)
            return_levels = np.full((len(rps), len(epochs), h, w), np.nan)
            betas0 = np.full((h, w), np.nan)
            betas1 = np.full((h, w), np.nan)

            i = 0

            for lat in range(h):
                for lon in range(w):


                    data = x[:, lat, lon]
                    if (data == -9999).all() or np.isnan(data).all():
                        continue
                    
                    data = data[data != -9999]
                    data = data[~np.isnan(data)]

                    years_valid = years[data != -9999]
                    years_valid = years_valid[~np.isnan(data)]

                    fit_poi = minimize(stats.nll_binom, [0, 0], args=(data, years_valid))
                    if not fit_poi.success:
                        warn(f"Trend fitting failed at lat {lat}, lon {lon}: {fit_poi.message}")
                        continue

                    beta0_trend, beta1_trend = fit_poi.x
                    print(f"Intercept: {beta0_trend:.4f}, Slope: {beta1_trend:.4f}")
                    
                    mean_rates = stats.mean_binom(years_valid, (beta0_trend, beta1_trend))
                    residuals = data - mean_rates

                    if dry_run:
                        if i % 100 == 0:
                            fig, ax = plt.subplots(figsize=(3, 2))
                            ax.plot(years_valid, data, color="blue")
                            ax.plot(years_valid, mean_rates, color='red')
                            ax.plot(years_valid, residuals, 'purple')
                            ax.set_title(f"hd{temperature} at lat {lat}, lon {lon}")
                            plt.show()
                    

                    # threshold = np.quantile(residuals, 0.9)
                    # select the top 32 exceedances
                    if len(residuals) < 10:
                        warn(f"Not enough data at lat {lat}, lon {lon}")
                        continue

                    threshold = 0.0
                    print(f"Chosen threshold at lat {lat}, lon {lon}: {threshold}")
                    exceedances = residuals[residuals > threshold] - threshold
                    exceed_years = years_valid[residuals > threshold]

                    if len(exceedances) < 10:
                        warn(f"Not enough exceedances at lat {lat}, lon {lon}")
                        continue
                    else:
                        print(f"Number of exceedances at lat {lat}, lon {lon}: {len(exceedances)}")
                    
                    # initial estimates using normal GPD
                    init_params = genpareto.fit(exceedances, floc=0)
                    shape_init = init_params[0]
                    scale_init = init_params[2]
                    init_params = [np.log(scale_init), 0, shape_init] # for nonstationary
                    
                    if scale_init <= 0:
                        warn(f"Stationary scale is non-positive at lat {lat}, lon {lon}")
                        continue

                    
                    result = minimize(
                        nll_dgpd,
                        init_params,
                        args=(exceedances, exceed_years),
                        method='L-BFGS-B',
                        bounds=[(None, None), (None, None), (-0.5, 1.0)]
                    )
                    
                    if not result.success:
                        warn(f"Optimization failed at lat {lat}, lon {lon}: {result.message}")
                        continue
                    
                    beta0_scale, beta1_scale, shape = result.x
                    
                    tail_indices[lat, lon] = shape
                    betas0[lat, lon] = beta0_scale
                    betas1[lat, lon] = beta1_scale
                    locs[lat, lon] = threshold
                    
                    # get return levels 
                    for i_epoch, epoch in enumerate(epochs):
                        outfile = os.path.join(output_dir, model, scenario, f"{epoch}.nc")
                        os.makedirs(os.path.dirname(outfile), exist_ok=True)
                        epoch_rescaled = (epoch - years_min) / (years_max - years_min)
                        for i, T in enumerate(rps):
                            base_rate = stats.mean_binom(epoch_rescaled, (beta0_trend, beta1_trend))
                            return_level = base_rate + threshold + dgpd_quantile_nonstationary(1/T, beta0_scale, beta1_scale, shape, epoch_rescaled)
                            return_levels[i, i_epoch, lat, lon] = min(return_level, 365)

                    i += 1
            
            return_levels_da = xr.DataArray(
                return_levels,
                dims=["return_period", "epoch", "rlat", "rlon"],
                coords={
                    "return_period": rps,
                    "epoch": epochs,
                    "rlat": ds_epoch["rlat"],
                    "rlon": ds_epoch["rlon"],
                },
                name=f"hd{temperature}",
                attrs={
                    "units": "days",
                    "long_name": f"Return levels for annual number of days exceeding {temperature}°C",
                    "distribution": "Discrete Generalized Pareto Distribution (D-GPD)",
                },
            )

            # Save results
            return_levels_ds = return_levels_da.to_dataset(name=f"hd{temperature}")
            return_levels_ds["tail_index"] = (("rlat", "rlon"), tail_indices)
            return_levels_ds["beta0_scale"] = (("rlat", "rlon"), betas0)
            return_levels_ds["beta1_scale"] = (("rlat", "rlon"), betas1)
            return_levels_ds["threshold"] = (("rlat", "rlon"), locs)

            return_levels_ds.to_netcdf(outfile)

            print(f"Saved D-GPD return levels to {outfile}")

            # ---- Plot ----
            # view the latest one
            fig, axs = plt.subplots(1, 3, figsize=(25, 5), 
                                    subplot_kw={"projection": ccrs.PlateCarree()})

            vmin = return_levels_da.min().item()
            vmax = return_levels_da.max().item()

            return_levels_da.sel(epoch=2030, return_period=10).plot(cmap="YlOrRd", ax=axs[0], 
                                                        vmin=vmin, vmax=vmax)
            return_levels_da.sel(epoch=2030, return_period=100).plot(cmap="YlOrRd", ax=axs[1], 
                                                        vmin=vmin, vmax=vmax)
            return_levels_da.sel(epoch=2030, return_period=1000).plot(cmap="YlOrRd", ax=axs[2], 
                                                        vmin=vmin, vmax=vmax)

            for ax in axs:
                ax.add_feature(cfeature.COASTLINE)
                ax.set_title(ax.get_title(), fontsize=12)
            plt.show()


            if dry_run:
                break
        if dry_run:
            break
    
print(f"Finished processing d-GPD hazard maps for hd{temperature}.")

# %%
